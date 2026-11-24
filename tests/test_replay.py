import jax
import jax.numpy as jnp
import numpy as np

from f1sim import replay as rp

G = 0.9
C, T, X = rp.CONTINUE, rp.TERMINAL, rp.CUT


def window(rewards, ends, valid=None):
    r = jnp.asarray([rewards], jnp.float32)
    e = jnp.asarray([ends], jnp.int8)
    v = jnp.ones_like(e, bool) if valid is None else jnp.asarray([valid])
    ret, disc, k = rp.nstep(r, e, v, len(rewards) - 1, G)
    return float(ret[0]), float(disc[0]), int(k[0])


def test_nstep_full_window():
    ret, disc, k = window([1, 2, 3, 4], [C, C, C, C])
    assert np.isclose(ret, 1 + G * 2 + G**2 * 3) and np.isclose(disc, G**3) and k == 3


def test_nstep_terminal_stops_without_bootstrap():
    ret, disc, _ = window([1, 2, 3, 4], [C, T, C, C])
    assert np.isclose(ret, 1 + G * 2) and disc == 0.0
    ret, disc, _ = window([5, 2, 3, 4], [T, C, C, C])  # the first step itself ends the episode
    assert ret == 5 and disc == 0.0


def test_nstep_cut_bootstraps_from_the_cut_row():
    """A CUT row's own transition is unusable (the race restarted after it), but its input is the car's real state."""
    ret, disc, k = window([1, 2, 3, 4], [C, C, X, C])
    assert np.isclose(ret, 1 + G * 2) and np.isclose(disc, G**2) and k == 2


def test_nstep_invalid_row_means_no_bootstrap():
    ret, disc, _ = window([1, 2, 3, 4], [C, C, C, C], [True, True, False, True])
    assert np.isclose(ret, 1 + G * 2) and disc == 0.0


def filled(rows=32, slots=3, steps=50, n=3):
    buf = rp.new(rows, slots, 2, 1)
    rng = np.random.default_rng(0)
    for t in range(steps):
        valid = rng.random(slots) < 0.8
        end = np.where(rng.random(slots) < 0.1, T, np.where(rng.random(slots) < 0.1, X, C))
        obs = np.stack([np.full(slots, t), np.arange(slots)], axis=-1)
        buf = rp.add(buf, jnp.asarray(obs), jnp.zeros((slots, 1)), jnp.full(slots, float(t)), jnp.asarray(valid), jnp.asarray(end, jnp.int8))
    return buf


def test_ring_keeps_the_last_rows():
    buf = filled(rows=32, steps=50)
    assert int(buf.size) == 32 and int(buf.ptr) == 50 % 32
    assert sorted(np.asarray(buf.reward[:, 0]).astype(int).tolist()) == list(range(18, 50))


def test_samples_start_on_valid_steps_and_stay_inside_the_written_window():
    n = 3
    buf = filled(n=n)
    b = jax.jit(lambda buf, k: rp.sample(buf, k, 512, n, G))(buf, jax.random.PRNGKey(0))
    t0, slot = np.asarray(b.obs[:, 0]).astype(int), np.asarray(b.obs[:, 1]).astype(int)
    rows = np.asarray(buf.obs[..., 0]).astype(int)
    valid, end = np.asarray(buf.valid), np.asarray(buf.end)
    where = {(int(rows[r, s]), s): (valid[r, s], end[r, s]) for r in range(rows.shape[0]) for s in range(rows.shape[1])}
    assert all(where[(t, s)][0] and where[(t, s)][1] != X for t, s in zip(t0, slot))
    t_next = np.asarray(b.next_obs[:, 0]).astype(int)
    assert np.all((t_next - t0 >= 1) & (t_next - t0 <= n)) and np.all(t_next <= 49) and np.all(t0 >= 18)
    assert np.all(np.asarray(b.next_obs[:, 1]).astype(int) == slot)
    # the returns match a brute-force n-step from the stored rows
    for i in range(64):
        t, s = t0[i], slot[i]
        ret, disc, k = 0.0, 0.0, None
        for j in range(n + 1):
            v, e = where[(t + j, s)]
            if j and (e == X or not v):
                k, disc = j, (G**j if v else 0.0)
                break
            if j == n:
                k, disc = n, G**n
                break
            ret += G**j * (t + j)
            if e == T:
                k, disc = j + 1, 0.0
                break
        assert np.isclose(float(b.reward[i]), ret, rtol=1e-5) and np.isclose(float(b.discount[i]), disc, rtol=1e-5)


def test_a_batch_with_no_valid_step_is_flagged():
    buf = rp.new(16, 2, 3, 1)
    for _ in range(10):
        buf = rp.add(buf, jnp.ones((2, 3)), jnp.zeros((2, 1)), jnp.ones(2), jnp.zeros(2, bool), jnp.zeros(2, jnp.int8))
    assert not bool(rp.sample(buf, jax.random.PRNGKey(0), 8, 3, G).ok)
    buf = rp.add(buf, jnp.ones((2, 3)), jnp.zeros((2, 1)), jnp.ones(2), jnp.ones(2, bool), jnp.zeros(2, jnp.int8))
    for _ in range(4):
        buf = rp.add(buf, jnp.ones((2, 3)), jnp.zeros((2, 1)), jnp.ones(2), jnp.ones(2, bool), jnp.zeros(2, jnp.int8))
    assert bool(rp.sample(buf, jax.random.PRNGKey(0), 8, 3, G).ok)


def test_packed_inputs_round_trip():
    """8 bits over a fixed range (square-root scale where asked), float16 for the rest; values outside are clipped."""
    spec = [(0.0, 1.0, True), (-2.0, 2.0, False), None, (0.0, 1.0, False)]
    c = rp.make_codec(spec)
    x = jnp.asarray([[0.0, -2.0, 1.2345, 0.0], [0.005, 0.37, -7.5, 0.5], [1.0, 2.0, 3.0, 1.0], [2.0, -9.0, 0.1, -1.0]])
    y = np.asarray(rp.decode(c, rp.encode(c, x)))
    assert np.allclose(y[:, 2], np.asarray(x[:, 2].astype(jnp.float16), np.float32))  # float16, exact
    assert np.allclose(y[:3, 0], [0.0, 0.005, 1.0], atol=4e-4)  # near zero the root scale is fine
    assert abs(y[1, 1] - 0.37) <= 4.0 / 255 / 2 + 1e-6
    assert np.allclose(y[3, [0, 1, 3]], [1.0, -2.0, 0.0], atol=1e-6)  # clipped to the range
    assert rp.bytes_per_step(4, 1, c) == 3 + 2 + 2 + 6 and rp.bytes_per_step(4, 1) == 8 + 2 + 6


def test_a_packed_buffer_samples_decoded_inputs():
    spec = [(0.0, 1.0, False)] * 3 + [None]
    c = rp.make_codec(spec)
    buf = rp.new(16, 2, 4, 1, c)
    for t in range(16):
        obs = jnp.full((2, 4), t / 16.0)
        buf = rp.add(buf, obs, jnp.zeros((2, 1)), jnp.ones(2), jnp.ones(2, bool), jnp.zeros(2, jnp.int8))
    b = rp.sample(buf, jax.random.PRNGKey(0), 32, 1, 0.9)
    o = np.asarray(b.obs)
    assert o.dtype == np.float32 and np.allclose(o[:, :3], o[:, 3:4], atol=1 / 255)
    assert np.allclose(np.asarray(b.next_obs)[:, 3] - o[:, 3], 1 / 16.0)  # the next row


def test_driver_inputs_packing():
    from f1sim.drivers import env as de

    c = de.storage_codec()
    assert c.narrow.shape[0] + c.wide.shape[0] == de.OBS and c.wide.shape[0] == 14  # the car's state and the preview
    assert rp.bytes_per_step(de.OBS, de.ACTIONS, c) == 218  # 216 before the stuck timer's 2 inputs (2026-11-16), 214 before the blend line's (2026-11-12)

"""Checkpoint history: each stage keeps its last 10 evals and its last 5 bests (user, 2026-11-25)."""
import os

import jax.numpy as jnp

from f1sim import checkpoint as ck


def test_save_eval_keeps_the_last_evals_and_bests(tmp_path):
    bests = {0, 50, 150, 200, 300, 350, 500, 600}
    for k in range(15):
        n = k * 50
        ck.save_eval(tmp_path, n, {"w": jnp.full(3, float(n))}, {"env_steps": n}, n in bests)
    steps = lambda d, pre: sorted(int(p.stem.split("-")[1]) for p in (tmp_path / d).glob(f"{pre}-*.npz"))
    assert steps("evals", "eval") == [n * 50 for n in range(5, 15)]
    assert steps("bests", "best") == [200, 300, 350, 500, 600]
    assert ck.meta(tmp_path / "latest.npz")["env_steps"] == 700 and ck.meta(tmp_path / "best.npz")["env_steps"] == 600
    assert os.path.samefile(tmp_path / "latest.npz", tmp_path / "evals" / "eval-700.npz")  # a link, no extra disk
    ck.save(tmp_path / "latest.npz", {"w": jnp.zeros(3)}, {"env_steps": 999})  # the end-of-run save
    assert ck.meta(tmp_path / "evals" / "eval-700.npz")["env_steps"] == 700  # ... leaves the eval's own file alone

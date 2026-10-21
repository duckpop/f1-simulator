"""
TD3 on the GPU: networks, the replay buffer (on the GPU), the update block
captured as CUDA graphs, and checkpoints. Next to it the designer's double
DQN (a small network, trained on the CPU next to the season workers) and its
checkpoints. The only module (besides the entry points) that imports torch.
"""

import copy
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from racer.policy import designer_shapes, layer_shapes


# -------------------------------------------------------------------------
# Networks
# -------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, obs):
        return self.model(obs)

    def numpy_params(self):
        """Weights as float32 arrays in policy.layer_shapes order."""
        return [p.detach().float().cpu().numpy() for p in self.model.parameters()]


class TwinCritic(nn.Module):
    """Both TD3 critics in one module (one optimizer, one backward pass)."""

    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()

        def q_net():
            return nn.Sequential(nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))
        self.q1 = q_net()
        self.q2 = q_net()

    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, obs, action):
        return self.q1(torch.cat([obs, action], dim=1))


# -------------------------------------------------------------------------
# Replay buffer (on the GPU, plus a part in RAM)
# -------------------------------------------------------------------------

def ram_available():
    """Bytes of RAM the system can still hand out without swapping (MemAvailable), or None."""
    try:
        with open("/proc/meminfo") as f:
            return next((int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable:")), None)
    except OSError:
        return None


class ReplayBuffer:
    """n-step transitions (s, a, R, s', discount). The first `capacity` live
    in two GPU tensors: obs (2, capacity, obs_dim) float16 - states and next
    states (inputs are normalized to ~[-1, 1]) - and rest (capacity,
    action_dim + 2) float32 - action, reward, discount. With host_capacity >
    0 (CUDA only), up to that many more live in RAM, added GROW rows at a
    time only while ram_margin_gb of RAM stays available.

    One ring over both parts: the GPU part fills first, then the RAM part,
    then the oldest rows are overwritten. Sampling is uniform over all rows:
    the GPU rows are two gathers inside the (CUDA-graph) update block; before
    each block, stage() picks which rows of each critic step come from RAM
    (Binomial(batch, RAM rows / all rows) per step), gathers them on the host and
    copies them in, and the block mixes them in by mask. Never saved."""

    GROW = 131072                     # RAM rows added at a time (~73 MB at 133 inputs)

    def __init__(self, capacity, obs_dim, action_dim, device, chunk=256, host_capacity=0, ram_margin_gb=2.0,
                 batch=256, slots=1, seed=0):
        self.gpu_capacity = int(capacity)
        self.device = device
        self.action_dim = action_dim
        self.obs = torch.zeros((2, self.gpu_capacity, obs_dim), dtype=torch.float16, device=device)
        self.rest = torch.zeros((self.gpu_capacity, action_dim + 2), dtype=torch.float32, device=device)
        self.ptr = 0                      # next row of the ring (GPU rows first, then RAM rows)
        self.size = 0
        self.wrapped = False
        # Device-side copy of the GPU part's size, so sampling works inside a CUDA graph.
        self.size_t = torch.zeros((), dtype=torch.float32, device=device)
        pin = device.type == "cuda"
        self._host_s = torch.zeros((chunk, obs_dim), dtype=torch.float16, pin_memory=pin)
        self._host_ns = torch.zeros((chunk, obs_dim), dtype=torch.float16, pin_memory=pin)
        self._host_rest = torch.zeros((chunk, action_dim + 2), dtype=torch.float32, pin_memory=pin)
        self._copied = torch.cuda.Event() if pin else None

        # The RAM part. np.zeros maps its pages lazily: only rows written cost RAM.
        self.mixed = pin and host_capacity > 0
        self.host_max = int(host_capacity) if self.mixed else 0
        self.host_capacity = 0            # RAM rows usable so far
        self.margin = float(ram_margin_gb) * 1024 ** 3
        self.row_bytes = 2 * obs_dim * 2 + (action_dim + 2) * 4
        if self.mixed:
            self.ram_s = np.zeros((self.host_max, obs_dim), dtype=np.float16)
            self.ram_ns = np.zeros((self.host_max, obs_dim), dtype=np.float16)
            self.ram_rest = np.zeros((self.host_max, action_dim + 2), dtype=np.float32)
            # Per critic step of a block (slot): a batch of RAM rows on the GPU and which rows use them.
            self.slot_obs = torch.zeros((slots, 2, batch, obs_dim), dtype=torch.float16, device=device)
            self.slot_rest = torch.zeros((slots, batch, action_dim + 2), dtype=torch.float32, device=device)
            self.slot_mask = torch.zeros((slots, batch), dtype=torch.bool, device=device)
            # Two pinned staging sets: the host gathers the next block while the last copy is in flight.
            self._stage = [(torch.zeros((slots, 2, batch, obs_dim), dtype=torch.float16, pin_memory=True),
                            torch.zeros((slots, batch, action_dim + 2), dtype=torch.float32, pin_memory=True),
                            torch.zeros((slots, batch), dtype=torch.bool, pin_memory=True), torch.cuda.Event())
                           for _ in range(2)]
            self._stage_k = 0
            self._staged_once = [False, False]
            self.rng = np.random.default_rng(seed)

    @property
    def capacity(self):
        """Rows the ring holds now (the RAM part may still grow)."""
        return self.gpu_capacity + self.host_capacity

    @property
    def ram_rows(self):
        return max(0, self.size - self.gpu_capacity)

    def _grow(self):
        """Adds GROW rows to the RAM part if the ring hasn't wrapped yet and
        ram_margin_gb of RAM stays available afterwards."""
        if self.wrapped or self.host_capacity >= self.host_max:
            return False
        step = min(self.GROW, self.host_max - self.host_capacity)
        avail = ram_available()
        if avail is None or avail - step * self.row_bytes < self.margin:
            return False
        self.host_capacity += step
        return True

    def add_batch(self, s, a, r, ns, disc):
        """numpy arrays with a leading batch dim, written at the ring's
        pointer: GPU rows through a pinned staging buffer (one copy per
        tensor, without blocking), RAM rows directly."""
        n, k = len(s), 0
        while k < n:
            if self.ptr >= self.capacity and not self._grow():
                self.ptr, self.wrapped = 0, True
            if self.ptr < self.gpu_capacity:
                m = min(n - k, self.gpu_capacity - self.ptr, self._host_rest.shape[0])
                self._to_gpu(self.ptr, s[k:k + m], a[k:k + m], r[k:k + m], ns[k:k + m], disc[k:k + m])
            else:
                h = self.ptr - self.gpu_capacity
                m = min(n - k, self.host_capacity - h)
                self.ram_s[h:h + m] = s[k:k + m]
                self.ram_ns[h:h + m] = ns[k:k + m]
                self.ram_rest[h:h + m, :self.action_dim] = a[k:k + m]
                self.ram_rest[h:h + m, self.action_dim] = r[k:k + m]
                self.ram_rest[h:h + m, self.action_dim + 1] = disc[k:k + m]
            k += m
            self.ptr += m
            self.size = self.capacity if self.wrapped else max(self.size, self.ptr)
        self.size_t.fill_(float(min(self.size, self.gpu_capacity)))

    def _to_gpu(self, pos, s, a, r, ns, disc):
        n = len(s)
        if self._copied is not None:
            self._copied.synchronize()            # the last copy out of the staging buffer is done
        self._host_s[:n].numpy()[:] = s
        self._host_ns[:n].numpy()[:] = ns
        hr = self._host_rest[:n].numpy()
        hr[:, :self.action_dim] = a
        hr[:, self.action_dim] = r
        hr[:, self.action_dim + 1] = disc
        dst = slice(pos, pos + n)
        self.obs[0, dst].copy_(self._host_s[:n], non_blocking=True)
        self.obs[1, dst].copy_(self._host_ns[:n], non_blocking=True)
        self.rest[dst].copy_(self._host_rest[:n], non_blocking=True)
        if self._copied is not None:
            self._copied.record()

    def stage(self):
        """Before an update block: which rows of each critic step come from
        RAM, gathered and copied to the GPU (nothing while the RAM part is
        empty)."""
        ram = self.ram_rows
        if not self.mixed or ram == 0:
            return
        k = self._stage_k
        obs, rest, mask, done = self._stage[k]
        if self._staged_once[k]:
            done.synchronize()                    # this staging set's last copy has landed
        # Rows of a batch are interchangeable: the first Binomial(batch, RAM share)
        # of each come from RAM (contiguous copies), the rest from the GPU part.
        m, o, r = mask.numpy(), obs.numpy(), rest.numpy()
        for slot, count in enumerate(self.rng.binomial(m.shape[1], ram / self.size, size=m.shape[0])):
            pick = self.rng.integers(ram, size=count)
            m[slot, :count], m[slot, count:] = True, False
            o[slot, 0, :count] = self.ram_s[pick]
            o[slot, 1, :count] = self.ram_ns[pick]
            r[slot, :count] = self.ram_rest[pick]
        self.slot_obs.copy_(obs, non_blocking=True)
        self.slot_rest.copy_(rest, non_blocking=True)
        self.slot_mask.copy_(mask, non_blocking=True)
        done.record()
        self._staged_once[k] = True
        self._stage_k = 1 - k

    def sample(self, batch, slot=0):
        """A batch for critic step `slot` of the block (uniform over the GPU
        rows, with the staged RAM rows mixed in)."""
        idx = (torch.rand(batch, device=self.device) * self.size_t).long().clamp_(max=self.gpu_capacity - 1)
        obs, rest, a = self.obs[:, idx], self.rest[idx], self.action_dim
        if self.mixed:
            use = self.slot_mask[slot]
            obs = torch.where(use[None, :, None], self.slot_obs[slot], obs)
            rest = torch.where(use[:, None], self.slot_rest[slot], rest)
        obs = obs.float()
        return obs[0], rest[:, :a], rest[:, a:a + 1], obs[1], rest[:, a + 1:]

    def __len__(self):
        return self.size


# -------------------------------------------------------------------------
# TD3
# -------------------------------------------------------------------------

class TD3:
    def __init__(self, cfg, device):
        t = cfg.td3
        self.cfg = cfg
        self.device = device
        self.batch = t.batch
        self.tau = t.tau
        self.policy_noise = t.policy_noise
        self.noise_clip = t.noise_clip
        self.policy_freq = t.policy_freq
        self.total_it = 0
        self.memory = ReplayBuffer(t.replay_size, cfg.obs_dim, cfg.action_dim, device,
                                   host_capacity=t.host_replay_size, ram_margin_gb=t.host_replay_margin_gb,
                                   batch=t.batch, slots=t.policy_freq)
        self.actor = Actor(cfg.obs_dim, cfg.action_dim, t.hidden).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic = TwinCritic(cfg.obs_dim, cfg.action_dim, t.hidden).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        for p in list(self.target_actor.parameters()) + list(self.target_critic.parameters()):
            p.requires_grad_(False)
        on_cuda = device.type == "cuda"
        # capturable=True: the optimizer step can run inside a CUDA graph.
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=t.lr_actor, capturable=on_cuda)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=t.lr_critic, capturable=on_cuda)
        # Diagnostics (critic loss, actor loss, mean Q) stay on the GPU; they're
        # copied to pinned memory without blocking and read on the next call.
        self._diag = torch.zeros(3, device=device)
        self._diag_host = torch.zeros(3, pin_memory=on_cuda)
        self._diag_event = None
        self._diag_last = (None, None, None)
        self._have_diag = False
        self._graphs = {}
        self._use_graph = on_cuda
        self._actor_params = list(self.actor.parameters())
        self._critic_params = list(self.critic.parameters())

    # ---- update blocks -------------------------------------------------------
    def _critic_step(self, diag, slot):
        s, a, r, ns, disc = self.memory.sample(self.batch, slot)
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_a = (self.target_actor(ns) + noise).clamp(-1.0, 1.0)
            tq1, tq2 = self.target_critic(ns, next_a)
            target = r + disc * torch.min(tq1, tq2)
        q1, q2 = self.critic(s, a)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        if diag:
            with torch.no_grad():
                self._diag[0].copy_(loss)
                self._diag[2].copy_(q1.mean())
        return s

    def _actor_step(self, s):
        for p in self._critic_params:
            p.requires_grad_(False)
        loss = -self.critic.q1_only(s, self.actor(s)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()
        for p in self._critic_params:
            p.requires_grad_(True)
        with torch.no_grad():
            self._diag[1].copy_(loss)
            self._soft_update(self.actor, self.target_actor)
            self._soft_update(self.critic, self.target_critic)

    def _full_block(self):
        """policy_freq critic steps, then the delayed actor step + soft updates."""
        s = None
        for k in range(self.policy_freq):
            s = self._critic_step(diag=k == self.policy_freq - 1, slot=k)
        self._actor_step(s)

    def _critic_block(self):
        """Critic only (a resume's critic-only phase): the actor and its target stay put."""
        for k in range(self.policy_freq):
            self._critic_step(diag=k == self.policy_freq - 1, slot=k)
        with torch.no_grad():
            self._soft_update(self.critic, self.target_critic)

    def _soft_update(self, source, target):
        torch._foreach_lerp_(list(target.parameters()), list(source.parameters()), self.tau)

    def _capture(self, mode):
        """3 warm-up blocks on a side stream (they allocate grads and optimizer
        state; real updates), then capture one block."""
        block = self._full_block if mode == "full" else self._critic_block
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                block()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            block()
        self._graphs[mode] = graph
        return 3

    def train_block(self, update_actor=True):
        """One TD3 block (policy_freq critic updates). Returns the critic
        updates done (0 while the buffer is smaller than a batch)."""
        if len(self.memory) < self.batch:
            return 0
        mode = "full" if update_actor else "critic"
        block = self._full_block if update_actor else self._critic_block
        blocks = 1
        self.memory.stage()
        if self._use_graph and mode in self._graphs:
            self._graphs[mode].replay()
        elif self._use_graph:
            try:
                blocks = self._capture(mode)
            except Exception as e:                       # eager mode if capture isn't supported
                print(f"[TD3] CUDA graph capture failed ({e!r}); using eager updates.", flush=True)
                self._use_graph = False
                self._graphs = {}
                block()
        else:
            block()
        self._have_diag = True
        self.total_it += blocks * self.policy_freq
        return blocks * self.policy_freq

    # ---- learning rates, diagnostics -------------------------------------------
    def set_lr(self, actor=None, critic=None):
        """With CUDA graphs the lr is a GPU tensor the captured step reads on
        every replay; fill_() changes it in place."""
        for opt, lr in ((self.actor_opt, actor), (self.critic_opt, critic)):
            if lr is None:
                continue
            for g in opt.param_groups:
                if not self._use_graph:
                    g["lr"] = float(lr)
                elif torch.is_tensor(g["lr"]) and g["lr"].device.type == self.device.type:
                    g["lr"].fill_(float(lr))
                else:
                    g["lr"] = torch.tensor(float(lr), device=self.device)

    def diagnostics(self):
        """(critic loss, actor loss, mean Q) without blocking the GPU: the
        values copied on the previous call, once that copy has landed."""
        if not self._have_diag:
            return None, None, None
        if self.device.type != "cuda":
            return tuple(self._diag.tolist())
        if self._diag_event is None or self._diag_event.query():
            if self._diag_event is not None:
                self._diag_last = tuple(self._diag_host.tolist())
            self._diag_host.copy_(self._diag, non_blocking=True)
            self._diag_event = torch.cuda.Event()
            self._diag_event.record()
        return self._diag_last

    # ---- state -------------------------------------------------------------------
    def state_dicts(self):
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(), "target_critic": self.target_critic.state_dict(),
                "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
                "total_it": self.total_it}

    def load_state_dicts(self, d):
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        # The target actor restarts as the actor: a best checkpoint's networks
        # are from different moments (the evaluated actor, a newer critic).
        self.target_actor.load_state_dict(d["actor"])
        self.target_critic.load_state_dict(d["target_critic"])
        for key, opt in (("actor_opt", self.actor_opt), ("critic_opt", self.critic_opt)):
            if key in d:
                opt.load_state_dict(d[key])
        self.total_it = int(d.get("total_it", 0))


# -------------------------------------------------------------------------
# Checkpoints
# -------------------------------------------------------------------------

def save_checkpoint(agent, paths, meta=None, actor_params=None):
    """Writes the learner's networks to paths[0] and copies it to the rest.
    actor_params: the weights a worker evaluated, stored instead of the
    learner's current actor - without the actor's optimizer state, which
    belongs to the learner's newer actor."""
    cfg = agent.cfg
    model = agent.state_dicts()
    if actor_params is not None:
        model["actor"] = {name: torch.from_numpy(np.array(p, dtype=np.float32))
                          for name, p in zip(model["actor"], actor_params)}
        model.pop("actor_opt")
    first, *rest = paths
    first.parent.mkdir(parents=True, exist_ok=True)
    tmp = first.with_suffix(".tmp")
    torch.save({"model": model, "meta": meta or {}, "obs_dim": cfg.obs_dim, "action_dim": cfg.action_dim,
                "layout": cfg.layout}, tmp)
    tmp.replace(first)
    for path in rest:
        shutil.copyfile(first, path)


def load_checkpoint(path, cfg, map_location="cpu"):
    """The one checkpoint reader (learner, opponents, watcher, compare):
    (the whole saved dict, its networks). Refuses another layout."""
    data = torch.load(path, map_location=map_location, weights_only=True)
    if data.get("layout") != cfg.layout:
        raise SystemExit(f"'{path}' has layout {data.get('layout')!r}, this code needs {cfg.layout!r} - "
                         f"old checkpoints are never upgraded: train from scratch, or delete it.")
    return data, data["model"]


def read_actor_weights(path, cfg):
    """A checkpoint's actor as float32 numpy arrays in layer order (for the
    torch-free FrozenPolicy), and the saved dict."""
    data, model = load_checkpoint(path, cfg)
    params = [v.float().numpy() for v in model["actor"].values()]
    if [p.shape for p in params] != layer_shapes(cfg.obs_dim, cfg.action_dim, cfg.td3.hidden):
        raise SystemExit(f"'{path}' has a different network size ([td3] hidden changed?)")
    return params, data


# -------------------------------------------------------------------------
# The designer: double DQN
# -------------------------------------------------------------------------

class DesignerNet(nn.Module):
    """Designer inputs -> one Q-value per point operation (29 -> 128 -> 128 -> 26)."""

    def __init__(self, obs_dim, ops, hidden):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(inplace=True),
                                   nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, ops))

    def forward(self, obs):
        return self.model(obs)

    def numpy_params(self):
        """Weights as float32 arrays in policy.layer_shapes order."""
        return [p.detach().float().cpu().numpy() for p in self.model.parameters()]


class DesignerReplay:
    """(obs, operation, reward, next obs, discount, next obs' legal
    operations) on the CPU. Never saved."""

    def __init__(self, capacity, obs_dim, ops):
        self.capacity = int(capacity)
        self.obs = torch.zeros((self.capacity, obs_dim))
        self.next_obs = torch.zeros((self.capacity, obs_dim))
        self.op = torch.zeros((self.capacity, 1), dtype=torch.int64)
        self.reward = torch.zeros((self.capacity, 1))
        self.discount = torch.zeros((self.capacity, 1))
        self.next_mask = torch.zeros((self.capacity, ops), dtype=torch.bool)
        self.ptr = 0
        self.size = 0

    def add_batch(self, obs, op, reward, next_obs, discount, next_mask):
        for k in range(len(obs)):                  # a batch is one race's decisions: tens of rows
            i = self.ptr
            self.obs[i] = torch.from_numpy(np.asarray(obs[k], dtype=np.float32))
            self.next_obs[i] = torch.from_numpy(np.asarray(next_obs[k], dtype=np.float32))
            self.op[i, 0] = int(op[k])
            self.reward[i, 0] = float(reward[k])
            self.discount[i, 0] = float(discount[k])
            self.next_mask[i] = torch.from_numpy(np.asarray(next_mask[k], dtype=bool))
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch):
        idx = torch.randint(0, self.size, (batch,))
        return (self.obs[idx], self.op[idx], self.reward[idx], self.next_obs[idx], self.discount[idx],
                self.next_mask[idx])

    def __len__(self):
        return self.size


class DesignerDQN:
    """Double DQN: the online network picks the next state's best legal
    operation, the target network values it; Huber loss; the target network
    copied from the online one every target_every_updates."""

    def __init__(self, cfg, device=None):
        d = cfg.designer
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.batch = d.batch
        self.target_every = d.target_every_updates
        self.net = DesignerNet(cfg.designer_obs_dim, cfg.designer_ops, d.hidden).to(self.device)
        self.target = copy.deepcopy(self.net)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=d.lr)
        self.memory = DesignerReplay(d.replay_size, cfg.designer_obs_dim, cfg.designer_ops)
        self.total_it = 0
        self.last = (None, None)                   # (loss, mean Q) of the last update

    def update(self):
        """One gradient step; False while the buffer is smaller than a batch."""
        if len(self.memory) < self.batch:
            return False
        s, a, r, ns, disc, nmask = (x.to(self.device) for x in self.memory.sample(self.batch))
        with torch.no_grad():
            best = self.net(ns).masked_fill(~nmask, float("-inf")).argmax(dim=1, keepdim=True)
            target = r + disc * self.target(ns).gather(1, best)
        q = self.net(s).gather(1, a)
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.total_it += 1
        if self.total_it % self.target_every == 0:
            self.target.load_state_dict(self.net.state_dict())
        self.last = (loss.item(), q.mean().item())
        return True

    def state_dicts(self):
        return {"net": self.net.state_dict(), "target": self.target.state_dict(), "opt": self.opt.state_dict(),
                "total_it": self.total_it}

    def load_state_dicts(self, d):
        self.net.load_state_dict(d["net"])
        self.target.load_state_dict(d.get("target", d["net"]))
        if "opt" in d:
            self.opt.load_state_dict(d["opt"])
        self.total_it = int(d.get("total_it", 0))


def save_designer(dqn, paths, meta=None, params=None):
    """Writes the designer to paths[0] and copies it to the rest. params: the
    weights a test round evaluated, stored as the network instead of the
    learner's current one (without the optimizer state, which belongs to it)."""
    cfg = dqn.cfg
    model = dqn.state_dicts()
    if params is not None:
        model["net"] = {name: torch.from_numpy(np.array(p, dtype=np.float32)) for name, p in zip(model["net"], params)}
        model["target"] = dict(model["net"])
        model.pop("opt")
    first, *rest = paths
    first.parent.mkdir(parents=True, exist_ok=True)
    tmp = first.with_suffix(".tmp")
    torch.save({"model": model, "meta": meta or {}, "obs_dim": cfg.designer_obs_dim, "ops": cfg.designer_ops,
                "layout": cfg.designer_layout}, tmp)
    tmp.replace(first)
    for path in rest:
        shutil.copyfile(first, path)


def load_designer(path, cfg, map_location="cpu"):
    """(the whole saved dict, its networks); refuses another layout."""
    data = torch.load(path, map_location=map_location, weights_only=True)
    if data.get("layout") != cfg.designer_layout:
        raise SystemExit(f"'{path}' has layout {data.get('layout')!r}, this code needs {cfg.designer_layout!r} - "
                         f"train the designer again.")
    return data, data["model"]


def read_designer_weights(path, cfg):
    """A designer checkpoint's network as float32 numpy arrays in layer order
    (for the torch-free season workers), and the saved dict."""
    data, model = load_designer(path, cfg)
    params = [v.float().numpy() for v in model["net"].values()]
    if [p.shape for p in params] != designer_shapes(cfg):
        raise SystemExit(f"'{path}' has a different network size ([designer] hidden changed?)")
    return params, data



def read_season_drivers(cfg, spec=""):
    """The designer's drivers: 'middle,aggressive:latest,defensive' (default:
    every [season] profile's best) -> ({profile: actor weights}, ['profile:tag@updates'])."""
    items = [x.strip() for x in spec.split(",") if x.strip()] if spec else list(cfg.season.profiles)
    drivers, ids = {}, []
    for item in items:
        profile, _, tag = item.partition(":")
        if profile not in cfg.season.profiles:
            raise SystemExit(f"--drivers: {profile!r} isn't one of the [season] profiles "
                             f"({', '.join(cfg.season.profiles)})")
        if profile in drivers:
            raise SystemExit(f"--drivers: {profile!r} listed twice")
        path = cfg.checkpoint_path(profile, tag or "best")
        if not path.exists():
            raise SystemExit(f"--drivers: no checkpoint at '{path}' - train '{profile}' first "
                             f"(project_v2.md §8), or pick another tag")
        drivers[profile], data = read_actor_weights(path, cfg)
        ids.append(f"{profile}:{tag or 'best'}@{data['model'].get('total_it', '?')}")
    return drivers, ids

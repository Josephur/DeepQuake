"""Proximal Policy Optimisation for multi-discrete actions, with a GRU core.

Self-contained: a small recurrent actor-critic and the PPO update. It knows
nothing about Quake; it works with any vectorised environment that has
``n``, ``first_obs`` and ``step(actions)`` (see :mod:`deepquake.vec`).

The policy remembers: a GRU carries a hidden state from step to step, so a
dead end seen a second ago can steer the next decision. The state is zero at
the start of every episode, travels with the rollout, and the update runs
backpropagation through time over fixed-length chunks of that rollout, using
each chunk's recorded hidden state as its starting point.

Requires PyTorch (``uv sync --extra train``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class PPOConfig:
    steps_per_env: int = 256  # rollout length per environment per update
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256  # width of the GRU state and of the heads' input
    bptt: int = 32  # steps per backpropagation-through-time chunk


class ActorCritic(nn.Module):
    """GRU policy and value function with one categorical head per action.

    The recurrent state is :data:`PPOConfig.hidden` wide and starts at zero
    for every episode. All methods take and return it explicitly, so callers
    (rollout, evaluation, tests) control episode boundaries.
    """

    ARCH = "gru"

    def __init__(
        self,
        obs_dim: int,
        action_nvec: tuple[int, ...],
        hidden: int = 256,
        n_types: int | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_nvec = tuple(action_nvec)
        self.hidden = hidden
        self.n_types = n_types
        # A one-layer nn.GRU is the same cell as GRUCell, and unlike the cell
        # it accepts packed variable-length batches, which the update replays
        # in a single call (#20).
        self.gru = nn.GRU(obs_dim, hidden, batch_first=True)
        # One shared network for every monster type (#48): the trunk is
        # shared, the type comes in as a one-hot at the end of the
        # observation, and a small head per type turns the state into that
        # body's actions. With n_types unset (navigation, the player policy)
        # there is one head, exactly as before.
        if n_types is None:
            self.policy_head = nn.Linear(hidden, sum(self.action_nvec))
            nn.init.orthogonal_(self.policy_head.weight, 0.01)
            nn.init.zeros_(self.policy_head.bias)
        else:
            self.policy_heads = nn.ModuleList(
                nn.Linear(hidden, sum(self.action_nvec)) for _ in range(n_types)
            )
            for head in self.policy_heads:
                nn.init.orthogonal_(head.weight, 0.01)
                nn.init.zeros_(head.bias)
        self.value_head = nn.Linear(hidden, 1)

    def initial_hidden(self, n: int, device=None) -> torch.Tensor:
        """Zero state for ``n`` episodes (or fresh environments)."""
        return torch.zeros(n, self.hidden, device=device)

    def advance(self, obs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """One GRU step: the observation and previous state give the new state."""
        return self.gru(obs.unsqueeze(1), hidden.unsqueeze(0))[1].squeeze(0)

    def head_list(self) -> list[nn.Linear]:
        """The policy head or heads, in type order."""
        return list(self.policy_heads) if self.n_types is not None else [self.policy_head]

    def head_logits(self, hidden: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        """Per-head logits. With per-type heads, ``obs``'s one-hot picks one.

        Selection is a weighted sum over the heads' weights, not a gather:
        it is one tensor expression the ONNX exporter turns into plain
        multiply/add ops, and the exported graph keeps a single observation
        input, which is what the play client feeds.
        """
        if self.n_types is None:
            return self.policy_head(hidden)
        if obs is None:
            raise ValueError("an observation is required to choose between per-type heads")
        one_hot = obs[:, obs.shape[-1] - self.n_types :]
        weights = torch.stack([head.weight for head in self.policy_heads])
        biases = torch.stack([head.bias for head in self.policy_heads])
        selected = torch.einsum("bt,toh->boh", one_hot, weights)
        return torch.bmm(selected, hidden.unsqueeze(-1)).squeeze(-1) + one_hot @ biases

    def distributions(
        self, hidden: torch.Tensor, obs: torch.Tensor | None = None
    ) -> list[torch.distributions.Categorical]:
        logits = self.head_logits(hidden, obs)
        return [
            torch.distributions.Categorical(logits=part)
            for part in logits.split(self.action_nvec, dim=-1)
        ]

    def value(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden).squeeze(-1)

    def act(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        *,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ):
        """One step. Returns ``actions (n, heads), log_prob (n,), value (n,), new_hidden``.

        Sampling uses ``generator`` when given, so worker processes and
        evaluation share one reproducible stream instead of the global one.
        """
        hidden = self.advance(obs, hidden)
        dists = self.distributions(hidden, obs)
        if greedy:
            actions = torch.stack([d.probs.argmax(-1) for d in dists], dim=-1)
        else:
            actions = torch.stack(
                [torch.multinomial(d.probs, 1, generator=generator).squeeze(-1) for d in dists],
                dim=-1,
            )
        log_prob = torch.stack([d.log_prob(actions[:, i]) for i, d in enumerate(dists)]).sum(0)
        return actions, log_prob, self.value(hidden), hidden

    def evaluate(
        self, hidden: torch.Tensor, actions: torch.Tensor, obs: torch.Tensor | None = None
    ):
        """Log-probability, entropy and value of given actions, from a state."""
        dists = self.distributions(hidden, obs)
        log_prob = torch.stack([d.log_prob(actions[:, i]) for i, d in enumerate(dists)]).sum(0)
        entropy = torch.stack([d.entropy() for d in dists]).sum(0)
        return log_prob, entropy, self.value(hidden)

    def init_from(self, source: ActorCritic) -> None:
        """Warm start from another network: take its trunk, keep its behaviour.

        Every shared parameter is copied. When this network is wider (the
        type one-hot appended to the observation), the extra input columns
        are zeroed, so a one-hot through them contributes nothing and this
        network acts exactly like the source did — the new body starts from
        the old brain instead of from nothing. Each per-type head takes the
        source's head of the same index, or its only head when the source had
        one, so a second type also starts from the first type's policy.
        """
        with torch.no_grad():
            source_gru = dict(source.gru.named_parameters())
            for name, target in self.gru.named_parameters():
                origin = source_gru[name]
                if target.shape == origin.shape:
                    target.copy_(origin)
                elif name.startswith("weight_ih"):
                    columns = min(origin.shape[1], target.shape[1])
                    target.zero_()
                    target[:, :columns].copy_(origin[:, :columns])
                else:
                    raise ValueError(
                        f"cannot warm start {name}: {tuple(origin.shape)} -> {tuple(target.shape)}"
                    )
            origins = source.head_list()
            for index, head in enumerate(self.head_list()):
                head.load_state_dict(origins[min(index, len(origins) - 1)].state_dict())
            self.value_head.load_state_dict(source.value_head.state_dict())

    def spec(self) -> dict:
        """What is needed, besides the weights, to rebuild this network."""
        return {
            "arch": self.ARCH,
            "obs_dim": self.obs_dim,
            "action_nvec": self.action_nvec,
            "hidden": self.hidden,
            "n_types": self.n_types,
        }


def save_checkpoint(path, model: ActorCritic, optimizer, extra: dict) -> None:
    torch.save(
        {
            "spec": model.spec(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            **extra,
        },
        path,
    )


def load_model(path, device="cpu") -> tuple[ActorCritic, dict]:
    """Rebuild a network from a checkpoint. Returns the model and the checkpoint dict."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    spec = dict(checkpoint["spec"])
    if spec.get("arch") != ActorCritic.ARCH:
        raise ValueError(
            f"{path} was trained by a {spec.get('arch', 'memoryless')} network; "
            f"this build uses {ActorCritic.ARCH!r}. Retrain (the observation layout "
            "is unchanged, the network is not)."
        )
    spec.pop("arch")
    model = ActorCritic(**spec).to(device)
    state = checkpoint["model"]
    if "gru.weight_ih_l0" not in state:  # GRUCell-era checkpoint: same cell, other names
        state = {
            key.replace("gru.weight_ih", "gru.weight_ih_l0")
            .replace("gru.weight_hh", "gru.weight_hh_l0")
            .replace("gru.bias_ih", "gru.bias_ih_l0")
            .replace("gru.bias_hh", "gru.bias_hh_l0"): value
            for key, value in state.items()
        }
    model.load_state_dict(state)
    return model, checkpoint


def chunk_segments(done: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split chunks at episode ends into run-length segments.

    ``done`` is ``(sequences, steps)`` with 1 where an episode ended. A done
    cuts *after* its step: the next step starts from a zero state. Returns
    ``(owner, start, length)`` — for each segment, the chunk it belongs to,
    the step it starts at, and how many steps it spans.
    """
    sequences, steps = done.shape
    ends = done.cpu().numpy()
    owner, start, length = [], [], []
    for b in range(sequences):
        s = 0
        for t in np.flatnonzero(ends[b]):
            owner.append(b)
            start.append(s)
            length.append(int(t) + 1 - s)
            s = int(t) + 1
        if s < steps:
            owner.append(b)
            start.append(s)
            length.append(steps - s)
    make = lambda values: torch.tensor(values, dtype=torch.long)  # noqa: E731
    return make(owner), make(start), make(length)


def packed_replay(
    model: ActorCritic,
    obs_seq: torch.Tensor,
    act_seq: torch.Tensor,
    hid_seq: torch.Tensor,
    owner: torch.Tensor,
    start: torch.Tensor,
    seg_len: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Replay segments through the GRU in one packed pass.

    Returns ``(owner, time, log_prob, entropy, value)`` flattened over all
    real (non-padded) segment positions, plus the gather indices, so callers
    can align any per-step buffer with ``buffer[owner, time]``. A segment
    starting at step 0 begins from the chunk's recorded hidden state
    (``hid_seq``); every later segment starts from zeros, exactly as the
    rollout zeroed the state at an episode end.
    """
    device = obs_seq.device
    seg_len_cpu = seg_len.cpu()
    maxlen = int(seg_len_cpu.max())
    steps = obs_seq.shape[1]
    time = start.to(device).unsqueeze(1) + torch.arange(maxlen, device=device)
    # Padded positions may index past the chunk end; they are masked out
    # below, so pin them to the last step to keep the gather in range.
    time = torch.minimum(time, torch.full_like(time, steps - 1))
    valid = torch.arange(maxlen, device=device) < seg_len_cpu.to(device).unsqueeze(1)
    obs = obs_seq[owner.to(device).unsqueeze(1), time]  # (segments, maxlen, obs)
    actions = act_seq[owner.to(device).unsqueeze(1), time]
    hidden0 = torch.zeros(len(seg_len), model.hidden, device=device)
    first = start.to(device) == 0
    hidden0[first] = hid_seq[owner.to(device)[first]]
    packed = nn.utils.rnn.pack_padded_sequence(obs, seg_len_cpu, batch_first=True,
                                           enforce_sorted=False)  # fmt: skip
    out, _ = model.gru(packed, hidden0.unsqueeze(0))
    out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    valid = valid[:, : out.shape[1]]
    log_prob, entropy, value = model.evaluate(out[valid], actions[valid], obs[valid])
    flat_owner = owner.to(device).unsqueeze(1).expand_as(valid)[valid]
    flat_time = time[:, : out.shape[1]][valid]
    return flat_owner, flat_time, log_prob, entropy, value


class RolloutActor:
    """A policy that acts inside an environment worker (#20).

    Holds a CPU copy of the network, a seeded sampling generator, and the
    environment's recurrent state — which persists across :meth:`sync` calls
    and is zeroed only when an episode ends, exactly as it was when the
    learner stepped the environments itself. ``rollout`` runs ``steps``
    decisions locally and returns the whole buffer, so training costs one
    message per update instead of two per decision.
    """

    def __init__(self, spec: dict, state: dict, seed: int):
        spec = dict(spec)
        spec.pop("arch", None)
        # One thread per worker: the model is tiny, and twenty workers each
        # spawning a full thread pool fight over the cores instead (#20).
        torch.set_num_threads(1)
        self.model = ActorCritic(**spec)
        self.model.eval()
        self.sync(state, seed)
        self.hidden = self.model.initial_hidden(1)

    def sync(self, state: dict, seed: int) -> None:
        """Take over new weights, and reseed the sampling stream."""
        self.model.load_state_dict(state)
        self.generator = torch.Generator().manual_seed(seed)

    def rollout(self, env, obs, steps: int) -> dict:
        """Run ``steps`` decisions from the worker's current ``obs``.

        The caller keeps ``next_obs`` from the result as its current
        observation; the episode (and the memory) continues across calls.
        """
        rows: dict[str, list] = {
            key: [] for key in ("obs", "act", "logp", "val", "rew", "done", "hid")
        }
        episodes = successes = 0
        for _ in range(steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                actions, log_prob, value, hidden = self.model.act(
                    obs_t, self.hidden, generator=self.generator
                )
            rows["hid"].append(self.hidden[0].numpy())
            result = env.step(actions[0].numpy())
            done = bool(result.done)
            obs = env.reset() if done else result.obs
            rows["obs"].append(np.asarray(obs, dtype=np.float32))
            rows["act"].append(actions[0].numpy())
            rows["logp"].append(log_prob[0].item())
            rows["val"].append(value[0].item())
            rows["rew"].append(float(result.reward))
            rows["done"].append(done)
            episodes += done
            successes += bool(result.success)
            # An episode boundary resets memory: the next episode starts blind.
            self.hidden = hidden * (1.0 - float(done))
        return {
            **{key: np.stack(values) for key, values in rows.items()},
            "episodes": episodes,
            "successes": successes,
            "next_obs": np.asarray(obs, dtype=np.float32),
            "next_hidden": self.hidden[0].numpy(),
        }


class PPO:
    """Collects rollouts from a vectorised environment and updates the network."""

    def __init__(self, env, model: ActorCritic, config: PPOConfig, device, seed: int = 0):
        if config.steps_per_env % config.bptt:
            raise ValueError(
                f"steps_per_env ({config.steps_per_env}) must be a multiple of bptt ({config.bptt})"
            )
        self.env = env
        self.model = model
        self.config = config
        self.device = device
        self.rollout_seed = seed  # sync() reseeds workers from it each update
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.total_steps = 0

    def update(self) -> dict:
        """One rollout and one PPO update. Returns statistics for logging."""
        cfg, n, length = self.config, self.env.n, self.config.bptt
        steps = cfg.steps_per_env
        device = self.device

        # The rollout runs inside the workers, on their own CPU copy of the
        # policy (#20): pushing the weights and reading the whole buffer back
        # costs one round trip per update, not two per decision.
        state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
        self.env.sync(self.model.spec(), state, self.rollout_seed)
        self.rollout_seed += n
        data = self.env.rollout(steps)
        self.total_steps += steps * n

        to = lambda key, dtype=torch.float32: torch.as_tensor(data[key], dtype=dtype, device=device)  # noqa: E731
        obs_buf = to("obs")
        hid_buf = to("hid")
        act_buf = to("act", torch.long)
        logp_buf = to("logp")
        val_buf = to("val")
        rew_buf = to("rew")
        done_buf = to("done")
        episodes, successes = int(data["episodes"]), int(data["successes"])

        # Generalised advantage estimation. An episode end cuts the bootstrap.
        with torch.no_grad():
            advantages = torch.zeros_like(rew_buf)
            last = torch.zeros(n, device=device)
            # The value of the last observation: advance the memory over it
            # (a finished episode's state is zero, so this is its fresh start).
            next_obs = to("next_obs")
            next_hidden = to("next_hidden")
            next_value = self.model.value(self.model.advance(next_obs, next_hidden))
            for t in reversed(range(steps)):
                not_done = 1.0 - done_buf[t]
                delta = rew_buf[t] + cfg.gamma * next_value * not_done - val_buf[t]
                last = delta + cfg.gamma * cfg.gae_lambda * not_done * last
                advantages[t] = last
                next_value = val_buf[t]
        returns = advantages + val_buf

        # Split the rollout into fixed-length sequences: one environment's run
        # of ``bptt`` steps starting from a recorded hidden state. The update
        # replays each sequence through the GRU so gradients reach the memory,
        # and zeroes the state where an episode ended inside the sequence.
        chunks = steps // length
        obs_seq = (
            obs_buf.reshape(chunks, length, n, -1)
            .permute(2, 0, 1, 3)
            .reshape(n * chunks, length, -1)
        )
        act_seq = (
            act_buf.reshape(chunks, length, n, -1)
            .permute(2, 0, 1, 3)
            .reshape(n * chunks, length, -1)
        )
        logp_seq = logp_buf.reshape(chunks, length, n).permute(2, 0, 1).reshape(n * chunks, length)
        adv_seq = advantages.reshape(chunks, length, n).permute(2, 0, 1).reshape(n * chunks, length)
        ret_seq = returns.reshape(chunks, length, n).permute(2, 0, 1).reshape(n * chunks, length)
        done_seq = done_buf.reshape(chunks, length, n).permute(2, 0, 1).reshape(n * chunks, length)
        hid_seq = (
            hid_buf.reshape(chunks, length, n, -1)[:, 0].permute(1, 0, 2).reshape(n * chunks, -1)
        )

        # Episode ends inside a chunk cut it into segments, each replayed from
        # a known state: the chunk's recorded hidden state, or zeros after a
        # cut. Packed into one nn.GRU call per minibatch, the replay is one
        # pass over the batch instead of a Python loop per timestep (#20).
        seg_owner, seg_start, seg_len = chunk_segments(done_seq)
        batch = n * chunks
        stats = {}
        for _ in range(cfg.epochs):
            for index in torch.randperm(batch, device=device).chunk(cfg.minibatches):
                pick = torch.isin(seg_owner, index.cpu())
                owner_f, time_f, log_prob, entropy, value = packed_replay(
                    self.model,
                    obs_seq,
                    act_seq,
                    hid_seq,
                    seg_owner[pick],
                    seg_start[pick],
                    seg_len[pick],
                )

                def flat(x, owner_f=owner_f, time_f=time_f):
                    """Any per-step buffer (sequences, steps) at the replayed positions."""
                    return x[owner_f, time_f]

                adv = flat(adv_seq)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = (log_prob - flat(logp_seq)).exp()
                policy_loss = -torch.min(
                    ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv
                ).mean()
                value_loss = 0.5 * (value - flat(ret_seq)).pow(2).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                stats = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.mean().item(),
                }
        stats.update(
            total_steps=self.total_steps,
            episodes=episodes,
            successes=successes,
            success_rate=successes / episodes if episodes else float("nan"),
            mean_reward=rew_buf.mean().item(),
        )
        return stats


def config_from_dict(values: dict) -> PPOConfig:
    """Build a config, rejecting unknown keys so typos in config files fail loudly."""
    known = asdict(PPOConfig())
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown PPO settings: {sorted(unknown)}")
    return PPOConfig(**values)


def pick_device(name_contains: str | None = None) -> torch.device:
    """CUDA device whose name contains the given text, else any CUDA device, else CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if name_contains:
        for index in range(torch.cuda.device_count()):
            if name_contains.lower() in torch.cuda.get_device_name(index).lower():
                return torch.device(f"cuda:{index}")
        raise RuntimeError(f"no CUDA device with a name containing {name_contains!r}")
    return torch.device("cuda:0")


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

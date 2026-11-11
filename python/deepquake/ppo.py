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

    def __init__(self, obs_dim: int, action_nvec: tuple[int, ...], hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_nvec = tuple(action_nvec)
        self.hidden = hidden
        self.gru = nn.GRUCell(obs_dim, hidden)
        self.policy_head = nn.Linear(hidden, sum(self.action_nvec))
        self.value_head = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.policy_head.weight, 0.01)
        nn.init.zeros_(self.policy_head.bias)

    def initial_hidden(self, n: int, device=None) -> torch.Tensor:
        """Zero state for ``n`` episodes (or fresh environments)."""
        return torch.zeros(n, self.hidden, device=device)

    def advance(self, obs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """One GRU step: the observation and previous state give the new state."""
        return self.gru(obs, hidden)

    def distributions(self, hidden: torch.Tensor) -> list[torch.distributions.Categorical]:
        logits = self.policy_head(hidden)
        return [
            torch.distributions.Categorical(logits=part)
            for part in logits.split(self.action_nvec, dim=-1)
        ]

    def value(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden).squeeze(-1)

    def act(self, obs: torch.Tensor, hidden: torch.Tensor, *, greedy: bool = False):
        """One step. Returns ``actions (n, heads), log_prob (n,), value (n,), new_hidden``."""
        hidden = self.advance(obs, hidden)
        dists = self.distributions(hidden)
        if greedy:
            actions = torch.stack([d.probs.argmax(-1) for d in dists], dim=-1)
        else:
            actions = torch.stack([d.sample() for d in dists], dim=-1)
        log_prob = torch.stack([d.log_prob(actions[:, i]) for i, d in enumerate(dists)]).sum(0)
        return actions, log_prob, self.value(hidden), hidden

    def evaluate(self, hidden: torch.Tensor, actions: torch.Tensor):
        """Log-probability, entropy and value of given actions, from a state."""
        dists = self.distributions(hidden)
        log_prob = torch.stack([d.log_prob(actions[:, i]) for i, d in enumerate(dists)]).sum(0)
        entropy = torch.stack([d.entropy() for d in dists]).sum(0)
        return log_prob, entropy, self.value(hidden)

    def spec(self) -> dict:
        """What is needed, besides the weights, to rebuild this network."""
        return {
            "arch": self.ARCH,
            "obs_dim": self.obs_dim,
            "action_nvec": self.action_nvec,
            "hidden": self.hidden,
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
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


class PPO:
    """Collects rollouts from a vectorised environment and updates the network."""

    def __init__(self, env, model: ActorCritic, config: PPOConfig, device):
        if config.steps_per_env % config.bptt:
            raise ValueError(
                f"steps_per_env ({config.steps_per_env}) must be a multiple of bptt ({config.bptt})"
            )
        self.env = env
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.obs = torch.as_tensor(env.first_obs, dtype=torch.float32, device=device)
        self.hidden = model.initial_hidden(env.n, device=device)
        self.total_steps = 0

    def update(self) -> dict:
        """One rollout and one PPO update. Returns statistics for logging."""
        cfg, n, length = self.config, self.env.n, self.config.bptt
        steps = cfg.steps_per_env
        device = self.device
        obs_buf = torch.zeros((steps, n, self.model.obs_dim), device=device)
        hid_buf = torch.zeros((steps, n, self.model.hidden), device=device)
        act_buf = torch.zeros(
            (steps, n, len(self.model.action_nvec)), dtype=torch.long, device=device
        )
        logp_buf = torch.zeros((steps, n), device=device)
        val_buf = torch.zeros((steps, n), device=device)
        rew_buf = torch.zeros((steps, n), device=device)
        done_buf = torch.zeros((steps, n), device=device)
        episodes = successes = 0

        for t in range(steps):
            with torch.no_grad():
                actions, log_prob, value, hidden = self.model.act(self.obs, self.hidden)
            hid_buf[t] = self.hidden  # the state this decision was made from
            next_obs, reward, done, success = self.env.step(actions.cpu().numpy())
            obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = self.obs, actions, log_prob, value
            rew_buf[t] = torch.as_tensor(reward, device=device)
            done_buf[t] = torch.as_tensor(done, dtype=torch.float32, device=device)
            episodes += int(done.sum())
            successes += int(success.sum())
            # An episode boundary resets memory: the next episode starts blind.
            self.hidden = hidden * (1.0 - done_buf[t].unsqueeze(-1))
            self.obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        self.total_steps += steps * n

        # Generalised advantage estimation. An episode end cuts the bootstrap.
        with torch.no_grad():
            advantages = torch.zeros_like(rew_buf)
            last = torch.zeros(n, device=device)
            # The value of the last observation: advance the memory over it
            # (a finished episode's state is zero, so this is its fresh start).
            next_value = self.model.value(self.model.advance(self.obs, self.hidden))
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

        def replay(index):
            """Log-prob, entropy, values for whole sequences, stepped through the GRU."""
            hidden = hid_seq[index]
            logs, entropies, values = [], [], []
            for t in range(length):
                hidden = self.model.advance(obs_seq[index, t], hidden)
                log_prob, entropy, value = self.model.evaluate(hidden, act_seq[index, t])
                logs.append(log_prob)
                entropies.append(entropy)
                values.append(value)
                # A sequence can contain an episode boundary; the next step
                # then starts from a zero state, exactly as in the rollout.
                hidden = hidden * (1.0 - done_seq[index, t].unsqueeze(-1))
            return (
                torch.stack(logs).transpose(0, 1),
                torch.stack(entropies).transpose(0, 1),
                torch.stack(values).transpose(0, 1),
            )

        batch = n * chunks
        stats = {}
        for _ in range(cfg.epochs):
            for index in torch.randperm(batch, device=device).chunk(cfg.minibatches):
                log_prob, entropy, value = replay(index)
                adv = adv_seq[index]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = (log_prob - logp_seq[index]).exp()
                policy_loss = -torch.min(
                    ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv
                ).mean()
                value_loss = 0.5 * (value - ret_seq[index]).pow(2).mean()
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

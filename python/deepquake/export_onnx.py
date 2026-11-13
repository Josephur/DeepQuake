"""Export a trained policy checkpoint to ONNX for the client (#41, M5).

    uv run python -m deepquake.export_onnx runs/grunt_pool/checkpoint.pt models/grunt.onnx

The exported graph takes an observation batch and the recurrent state and
returns, per decision:

- ``action_probabilities`` — the per-head softmax probabilities,
  concatenated in :attr:`ppo.ActorCritic` head order (split by
  ``ACTION_NVEC``/the manifest's layout identity and pick per head),
- ``value`` — the critic's state value,
- ``hidden_out`` — the recurrent state to carry into the next decision.

The batch axis is dynamic, so one process can later batch every
policy-driven monster of a tick into one inference. The head layout is
deliberately *not* stored in the file: the model manifest (#45) carries
the observation/action layout identity so a stale model refuses to load.

Two sidecars sit next to the model: ``<model>.nvec`` (action-head sizes,
then the turn grid) and, for a monster policy, ``<model>.monsters`` —
one ``<classname> <type index>`` per line, the class a shared model
drives and the one-hot slot its observation ends with (#48). A player
policy has no class map. A shared model's parity batch uses real
one-hots, one body per row, so the check exercises the per-type heads.

Verification: the export runs an onnxruntime parity check inline —
seeded observations through both the PyTorch policy and the exported
graph, greedy actions and probabilities compared — and the test suite
repeats it (synthetic weights in CI, real checkpoints where they exist).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from torch import nn

from . import ppo


class _PolicyGraph(nn.Module):
    """The export view of :class:`ppo.ActorCritic`: probabilities out, state carried."""

    def __init__(self, model: ppo.ActorCritic):
        super().__init__()
        self.model = model

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor):
        new_hidden = self.model.gru(obs.unsqueeze(1), hidden.unsqueeze(0))[1].squeeze(0)
        logits = self.model.head_logits(new_hidden, obs)
        probs = torch.cat(
            [torch.softmax(part, dim=-1) for part in logits.split(self.model.action_nvec, dim=-1)],
            dim=-1,
        )
        value = self.model.value_head(new_hidden).squeeze(-1)
        return probs, value, new_hidden


def export_onnx(checkpoint_path: Path, out_path: Path, *, samples: int = 64, seed: int = 0) -> dict:
    """Export ``checkpoint_path`` to ``out_path`` and verify parity with onnxruntime.

    Returns the spec and the measured maximum probability deviation, for
    the caller to record.
    """
    model, checkpoint = ppo.load_model(checkpoint_path, device="cpu")
    model.eval()
    wrapper = _PolicyGraph(model)
    obs_dim = model.obs_dim

    torch.onnx.export(
        wrapper,
        (torch.zeros(1, obs_dim), torch.zeros(1, model.hidden)),
        str(out_path),
        input_names=["observation", "hidden_in"],
        output_names=["action_probabilities", "value", "hidden_out"],
        dynamic_axes={
            "observation": {0: "batch"},
            "hidden_in": {0: "batch"},
            "action_probabilities": {0: "batch"},
            "value": {0: "batch"},
            "hidden_out": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,  # the TorchScript exporter: no onnxscript dependency
    )

    # Parity: the same seeded inputs through both implementations. For a
    # shared model the type one-hot must be a real one-hot, so each row is
    # one body rather than a mixture of heads.
    rng = np.random.default_rng(seed)
    batch = rng.standard_normal((samples, obs_dim)).astype(np.float32)
    if model.n_types is not None:
        batch[:, -model.n_types :] = 0.0
        for row in range(samples):
            batch[row, obs_dim - model.n_types + row % model.n_types] = 1.0
    hidden = rng.standard_normal((samples, model.hidden)).astype(np.float32)
    with torch.no_grad():
        probs_torch, _value_torch, hidden_torch = wrapper(
            torch.as_tensor(batch), torch.as_tensor(hidden)
        )
    session = onnxruntime.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    probs_onnx, _value_onnx, hidden_onnx = session.run(
        None,
        {"observation": batch, "hidden_in": hidden},
    )
    # The client's layout sidecar: the action-head sizes, one line, then
    # the turn grid the decision maps through (monster_env's, in degrees).
    from . import monster, monster_env

    sidecar = Path(str(out_path) + ".nvec")
    sidecar.write_text(
        " ".join(str(n) for n in model.action_nvec)
        + "\n"
        + " ".join(f"{d:g}" for d in monster_env.MONSTER_TURN_DEGREES)
        + "\n"
    )

    # Which monster classes this model drives, and which type index each one
    # is in the shared observation (deepquake.monster.MONSTER_TYPES). The run
    # records its classes in the checkpoint; the client reads this to route
    # edicts to the model and to write the type one-hot (#48). A model that is
    # not a monster policy (the player) gets no class map: nothing routes to
    # it by classname.
    declared = checkpoint.get("monsters")
    if declared is None and model.n_types is None:
        classes: list[str] | None = None
    elif declared is None:
        classes = list(monster.MONSTER_TYPES[: model.n_types])
    else:
        classes = list(declared)
        expected = model.n_types if model.n_types is not None else 1
        if len(classes) != expected:
            raise ValueError(
                f"{checkpoint_path}: the run names {len(classes)} monster classes "
                f"({classes}) but the network has {expected} type input(s)"
            )
    if classes is not None:
        try:
            indices = [(monster.MONSTER_TYPES.index(name), name) for name in classes]
        except ValueError as exc:
            raise ValueError(
                f"{checkpoint_path}: unknown monster class in {classes}; "
                f"known types are {monster.MONSTER_TYPES}"
            ) from exc
        monsters_path = Path(str(out_path) + ".monsters")
        monsters_path.write_text("".join(f"{name} {index}\n" for index, name in sorted(indices)))
        names = [name for _index, name in sorted(indices)]
    else:
        names = []

    deviation = float(np.max(np.abs(probs_onnx - probs_torch.numpy())))
    if deviation > 1e-5:
        raise RuntimeError(
            f"ONNX parity check failed: max probability deviation {deviation:.2e} > 1e-5"
        )
    hidden_dev = float(np.max(np.abs(hidden_onnx - hidden_torch.numpy())))
    if hidden_dev > 1e-4:
        raise RuntimeError(
            f"ONNX hidden-state parity check failed: max deviation {hidden_dev:.2e} > 1e-4"
        )
    return {
        "obs_dim": obs_dim,
        "action_nvec": list(model.action_nvec),
        "hidden": model.hidden,
        "n_types": model.n_types,
        "monsters": names,
        "total_steps": checkpoint.get("total_steps"),
        "max_probability_deviation": deviation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    spec = export_onnx(args.checkpoint, args.out)
    print(f"wrote {args.out}")
    print(
        f"obs_dim={spec['obs_dim']} action_nvec={spec['action_nvec']} "
        f"steps={spec['total_steps']} max_deviation={spec['max_probability_deviation']:.2e}"
    )


if __name__ == "__main__":
    main()

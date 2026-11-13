"""ONNX export parity (#41). The synthetic checks run in CI; the real
checkpoint checks skip cleanly when the local run directories (gitignored)
are absent."""

import numpy as np
import pytest
import torch

from deepquake import ppo

onnxruntime = pytest.importorskip("onnxruntime")
export = pytest.importorskip("deepquake.export_onnx")


def _trained_toy(
    seed: int,
    obs_dim: int = 8,
    nvec: tuple[int, ...] = (3, 4, 2),
    hidden: int = 16,
    n_types: int | None = None,
):
    """A small actor-critic with deterministic-but-not-trivial weights."""
    torch.manual_seed(seed)
    model = ppo.ActorCritic(obs_dim, nvec, hidden, n_types=n_types)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(torch.linspace(-0.5, 0.5, param.numel()).reshape(param.shape))
            if "bias" in name:
                param.mul_(3.0)
    model.eval()
    return model


def _parity(model, tmp_path, samples: int = 64, batch: int | None = None):
    path = tmp_path / "policy.onnx"
    wrapper = export._PolicyGraph(model)
    torch.onnx.export(
        wrapper,
        (torch.zeros(1, model.obs_dim), torch.zeros(1, model.hidden)),
        str(path),
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
    rows = batch if batch is not None else samples
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((rows, model.obs_dim)).astype(np.float32)
    hidden = rng.standard_normal((rows, model.hidden)).astype(np.float32)
    with torch.no_grad():
        probs, _value, new_hidden = wrapper(torch.as_tensor(obs), torch.as_tensor(hidden))
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got_probs, _got_value, got_hidden = session.run(None, {"observation": obs, "hidden_in": hidden})
    assert np.argmax(got_probs, axis=1).tolist() == np.argmax(probs.numpy(), axis=1).tolist()
    assert np.max(np.abs(got_probs - probs.numpy())) < 1e-5
    assert np.max(np.abs(got_hidden - new_hidden.numpy())) < 1e-4
    # Per-head probabilities each sum to one: the client splits by ACTION_NVEC.
    sizes = np.cumsum([0] + list(model.action_nvec))
    sums = [got_probs[:, a:b].sum(axis=1) for a, b in zip(sizes[:-1], sizes[1:], strict=True)]
    for head_sum in sums:
        assert np.allclose(head_sum, 1.0, atol=1e-5)


def test_export_parity_single_and_batched(tmp_path):
    model = _trained_toy(seed=41)
    _parity(model, tmp_path, batch=1)
    _parity(model, tmp_path, batch=7)  # the dynamic batch axis


def test_export_matches_head_order(tmp_path):
    """The concatenated probabilities keep head order: permuting one head's
    weights permutes the same slice of the output."""
    model = _trained_toy(seed=41, nvec=(2, 3))
    wrapper = export._PolicyGraph(model)
    obs = torch.zeros(1, model.obs_dim)
    hidden = torch.zeros(1, model.hidden)
    with torch.no_grad():
        probs = wrapper(obs, hidden)[0][0]
    slices = [probs[a:b] for a, b in ((0, 2), (2, 5))]
    uniform = (np.full(2, 0.5), np.full(3, 1.0 / 3))
    for head_slice, flat in zip(slices, uniform, strict=True):
        assert float(head_slice.sum()) == pytest.approx(1.0, abs=1e-5)
        # Trained (non-degenerate) weights: no head sits on the uniform
        # distribution, and the two heads differ from each other.
        assert not np.allclose(head_slice.numpy(), flat, atol=1e-3)


def test_export_shared_model_writes_the_class_map(tmp_path):
    """A two-type model exports the per-type heads and names its classes (#48)."""
    from deepquake import monster, monster_env

    model = _trained_toy(
        seed=48,
        obs_dim=monster.MONSTER_OBS_DIM,
        nvec=monster_env.MONSTER_ACTION_NVEC,
        n_types=monster.N_MONSTER_TYPES,
    )
    # The sidecar maps each class to its canonical type index, whatever order
    # the run listed them in.
    checkpoint = tmp_path / "shared.pt"
    ppo.save_checkpoint(
        checkpoint, model, None, {"total_steps": 5, "monsters": ["monster_dog", "monster_army"]}
    )
    out = tmp_path / "shared.onnx"
    spec = export.export_onnx(checkpoint, out, samples=16)
    assert spec["n_types"] == monster.N_MONSTER_TYPES
    assert (tmp_path / "shared.onnx.monsters").read_text() == "monster_army 0\nmonster_dog 1\n"

    # Parity with the type one-hot set: each row is one body, not a mixture.
    session = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(3)
    obs = rng.standard_normal((4, monster.MONSTER_OBS_DIM)).astype(np.float32)
    obs[:, monster.MONSTER_OBS_BASE_DIM :] = 0.0
    for row in range(4):
        obs[row, monster.MONSTER_OBS_BASE_DIM + row % monster.N_MONSTER_TYPES] = 1.0
    hidden = rng.standard_normal((4, model.hidden)).astype(np.float32)
    with torch.no_grad():
        probs, _value, _state = export._PolicyGraph(model)(
            torch.as_tensor(obs), torch.as_tensor(hidden)
        )
    got, _got_value, _got_state = session.run(None, {"observation": obs, "hidden_in": hidden})
    assert np.max(np.abs(got - probs.numpy())) < 1e-5

    # A run that names fewer classes than the model has types is a mistake.
    wrong = tmp_path / "wrong.pt"
    ppo.save_checkpoint(wrong, model, None, {"monsters": ["monster_army"]})
    with pytest.raises(ValueError, match="classes"):
        export.export_onnx(wrong, tmp_path / "wrong.onnx", samples=2)


CHECKPOINTS = [
    "runs/combat_grunt_fightstart/checkpoint.pt",  # the player policy (#38)
    "runs/grunt_pool/checkpoint.pt",  # the Grunt policy (#35)
]


@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_export_real_checkpoints(checkpoint, tmp_path):
    from pathlib import Path

    path = Path(checkpoint)
    if not path.exists():
        pytest.skip(f"{checkpoint} not present (gitignored run output)")
    model, _checkpoint = ppo.load_model(path, device="cpu")
    _parity(model, tmp_path, batch=5)

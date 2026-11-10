"""Training machinery on a toy environment: no simulator, game data or GPU needed."""

import numpy as np
import pytest

from deepquake import toy
from deepquake.vec import ProcVecEnv, WorkerError


class InProcessVecEnv:
    """The ProcVecEnv interface without processes, for fast PPO tests."""

    def __init__(self, n: int):
        self.envs = [toy.LineEnv(seed=i) for i in range(n)]
        self.n = n
        self.first_obs = np.stack([env.reset() for env in self.envs])

    def step(self, actions):
        rows = []
        for env, action in zip(self.envs, actions, strict=True):
            result = env.step(action)
            rows.append(
                (env.reset() if result.done else result.obs, result.reward, result.done,
                 result.success)
            )  # fmt: skip
        obs, reward, done, success = zip(*rows, strict=True)
        return np.stack(obs), np.array(reward, np.float32), np.array(done), np.array(success)


def test_proc_vec_env_steps_and_resets():
    with ProcVecEnv(toy.make_line_env, 3) as env:
        assert env.first_obs.shape == (3, toy.OBS_DIM)
        assert env.first_obs.dtype == np.float32
        dones = 0
        for _ in range(25):
            obs, reward, done, success = env.step(np.full((3, 1), 2))
            assert obs.shape == (3, toy.OBS_DIM) and reward.shape == (3,)
            dones += int(done.sum())
        assert dones >= 3  # every episode ends within 20 steps and restarts

        env.set_attr("scale", 0.0)  # start and target coincide from now on
        for _ in range(21):
            obs, _, done, success = env.step(np.full((3, 1), 1))
        assert np.allclose(obs, 0.0)


def test_proc_vec_env_matches_single_process():
    # Three steps: with seed 1 the episode ends, and the worker resets, on the fourth.
    reference = toy.LineEnv(seed=1)
    expected = [reference.reset()]
    for _ in range(3):
        expected.append(reference.step(np.array([2])).obs)
    with ProcVecEnv(toy.make_line_env, 2) as env:
        got = [env.first_obs[1]]
        for _ in range(3):
            got.append(env.step(np.full((2, 1), 2))[0][1])
    np.testing.assert_allclose(got, expected)


def _broken_env(index: int):
    raise ValueError("this environment cannot be built")


def test_worker_exception_reaches_the_parent():
    with pytest.raises(WorkerError, match="cannot be built"):
        ProcVecEnv(_broken_env, 1)


def test_ppo_learns_the_toy_task(tmp_path):
    torch = pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import ppo

    ppo.seed_everything(0)
    env = InProcessVecEnv(16)
    config = ppo.PPOConfig(steps_per_env=64, hidden=32, learning_rate=3e-3)
    model = ppo.ActorCritic(toy.OBS_DIM, toy.ACTION_NVEC, config.hidden)
    learner = ppo.PPO(env, model, config, torch.device("cpu"))

    first = learner.update()["success_rate"]
    for _ in range(25):
        stats = learner.update()
    assert stats["success_rate"] > 0.9, f"did not learn: {first:.2f} -> {stats['success_rate']:.2f}"
    assert stats["success_rate"] > first

    # A checkpoint restores a network that makes the same decisions.
    path = tmp_path / "checkpoint.pt"
    ppo.save_checkpoint(path, model, learner.optimizer, {"total_steps": learner.total_steps})
    restored, saved = ppo.load_model(path)
    assert saved["total_steps"] == learner.total_steps
    obs = torch.as_tensor(env.first_obs)
    assert torch.equal(model.act(obs, greedy=True)[0], restored.act(obs, greedy=True)[0])


def test_ppo_config_rejects_unknown_settings():
    pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import ppo

    with pytest.raises(ValueError, match="learning_rat"):
        ppo.config_from_dict({"learning_rat": 1e-3})

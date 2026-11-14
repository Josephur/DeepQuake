"""Training machinery on a toy environment: no simulator, game data or GPU needed."""

import numpy as np
import pytest

from deepquake import toy
from deepquake.vec import ProcVecEnv, WorkerError


class InProcessVecEnv:
    """The ProcVecEnv rollout interface without processes, for fast PPO tests.

    Mirrors what a vec.py worker does: one actor (policy copy plus seeded
    sampling) per environment, acting locally and returning whole rollouts.
    """

    def __init__(self, n: int):
        self.envs = [toy.LineEnv(seed=i) for i in range(n)]
        self.n = n
        self.obs = [env.reset() for env in self.envs]
        self.first_obs = np.stack(self.obs)
        self.actors: list = [None] * n

    def sync(self, spec, state, seed):
        from deepquake.ppo import RolloutActor

        for index in range(self.n):
            if self.actors[index] is None:
                self.actors[index] = RolloutActor(spec, state, seed + index)
            else:
                self.actors[index].sync(state, seed + index)

    def rollout(self, steps: int) -> dict:
        results = [
            self.actors[index].rollout(env, self.obs[index], steps)
            for index, env in enumerate(self.envs)
        ]
        self.obs = [result["next_obs"] for result in results]

        def per_step(key):
            stacked = np.stack([result[key] for result in results])
            return stacked.transpose(1, 0, *range(2, stacked.ndim))

        return {
            **{key: per_step(key) for key in ("obs", "hid", "act", "logp", "val", "rew", "done")},
            "episodes": sum(result["episodes"] for result in results),
            "successes": sum(result["successes"] for result in results),
            "next_obs": np.stack([result["next_obs"] for result in results]),
            "next_hidden": np.stack([result["next_hidden"] for result in results]),
        }


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
    hidden = model.initial_hidden(len(env.first_obs))
    first_actions = model.act(obs, hidden, greedy=True)
    again_actions = restored.act(obs, hidden, greedy=True)
    assert torch.equal(first_actions[0], again_actions[0])
    assert torch.equal(first_actions[3], again_actions[3])  # the memory agrees too


def test_the_recurrent_state_changes_the_decision():
    """The core must use its memory: the same view from two states differs."""
    torch = pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import ppo

    ppo.seed_everything(0)
    model = ppo.ActorCritic(toy.OBS_DIM, toy.ACTION_NVEC, hidden=16)
    obs = torch.zeros(1, toy.OBS_DIM)
    blind = model.initial_hidden(1)
    primed = torch.ones(1, model.hidden)
    logits_blind = model.distributions(model.advance(obs, blind))[0].logits
    logits_primed = model.distributions(model.advance(obs, primed))[0].logits
    assert not torch.allclose(logits_blind, logits_primed)

    # The state is carried into the next decision, not rebuilt from the view.
    _, _, _, state = model.act(obs, blind, greedy=True)
    assert not torch.allclose(state, blind)


def test_a_memoryless_checkpoint_is_refused(tmp_path):
    torch = pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import ppo

    path = tmp_path / "old.pt"
    torch.save(
        {
            "spec": {"obs_dim": toy.OBS_DIM, "action_nvec": toy.ACTION_NVEC, "hidden": 16},
            "model": {},
        },
        path,
    )
    with pytest.raises(ValueError, match="memoryless"):
        ppo.load_model(path)


def test_ppo_rejects_a_rollout_that_does_not_split_into_sequences():
    torch = pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import ppo

    env = InProcessVecEnv(2)
    config = ppo.PPOConfig(steps_per_env=10, bptt=4)
    model = ppo.ActorCritic(toy.OBS_DIM, toy.ACTION_NVEC, hidden=8)
    with pytest.raises(ValueError, match="multiple of"):
        ppo.PPO(env, model, config, torch.device("cpu"))


def test_ppo_config_rejects_unknown_settings():
    pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import ppo

    with pytest.raises(ValueError, match="learning_rat"):
        ppo.config_from_dict({"learning_rat": 1e-3})


def test_training_maps_must_be_disjoint_from_held_out_maps():
    from deepquake.train_nav import validate_maps

    maps, held_out = validate_maps({"maps": ["e1m1", "e1m2"], "held_out": ["e1m3"]})
    assert maps == ["e1m1", "e1m2"] and held_out == ["e1m3"]
    # The old single-map configs still work.
    assert validate_maps({"map": "e1m1"}) == (["e1m1"], [])
    with pytest.raises(ValueError, match="both trained on and held out"):
        validate_maps({"maps": ["e1m1"], "held_out": ["e1m1"]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_maps({"maps": ["e1m1", "e1m1"]})
    with pytest.raises(ValueError, match="at least one training map"):
        validate_maps({"maps": []})
    with pytest.raises(ValueError, match="needs 'maps'"):
        validate_maps({})
    # Every map needs a worker: the round-robin would otherwise skip it silently.
    with pytest.raises(ValueError, match="never trained on"):
        validate_maps({"maps": ["e1m1", "e1m2", "e1m3"], "envs": 2})
    assert validate_maps({"maps": ["e1m1", "e1m2"], "envs": 2}) == (["e1m1", "e1m2"], [])
    assert validate_maps({"maps": ["e1m1"], "envs": 4}) == (["e1m1"], [])  # maps repeat


def test_packed_replay_matches_step_by_step():
    """The packed one-pass replay computes what the timestep loop computed."""
    torch = pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import ppo

    ppo.seed_everything(1)
    model = ppo.ActorCritic(toy.OBS_DIM, toy.ACTION_NVEC, hidden=8)
    model.eval()
    sequences, steps, hidden = 5, 12, model.hidden
    obs_seq = torch.randn(sequences, steps, toy.OBS_DIM)
    act_seq = torch.stack(
        [torch.randint(0, n, (sequences, steps)) for n in toy.ACTION_NVEC], dim=-1
    )
    hid_seq = torch.randn(sequences, hidden)
    dones = (torch.rand(sequences, steps) < 0.25).float()

    owner, start, length = ppo.chunk_segments(dones)
    owner_f, time_f, logp, ent, val = ppo.packed_replay(
        model, obs_seq, act_seq, hid_seq, owner, start, length
    )

    # Segments must cover every (sequence, step) exactly once, like the loop.
    positions = sorted(zip(owner_f.tolist(), time_f.tolist(), strict=True))
    assert positions == sorted((b, t) for b in range(sequences) for t in range(steps))

    ref = torch.zeros(sequences, steps, 3)
    with torch.no_grad():
        for b in range(sequences):
            h = hid_seq[b]
            for t in range(steps):
                h = model.advance(obs_seq[b, t].unsqueeze(0), h.unsqueeze(0)).squeeze(0)
                one = model.evaluate(h, act_seq[b, t].unsqueeze(0))
                ref[b, t] = torch.stack([part.reshape(()) for part in one])
                h = h * (1.0 - dones[b, t])  # an episode end zeroes the state
    assert torch.allclose(logp, ref[owner_f, time_f, 0], atol=1e-5)
    assert torch.allclose(ent, ref[owner_f, time_f, 1], atol=1e-5)
    assert torch.allclose(val, ref[owner_f, time_f, 2], atol=1e-5)


def test_grucell_checkpoints_load_into_the_packed_model(tmp_path):
    """Checkpoints from the GRUCell build train on: same cell, other names."""
    torch = pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import ppo

    ppo.seed_everything(2)
    model = ppo.ActorCritic(toy.OBS_DIM, toy.ACTION_NVEC, hidden=8)
    old_names = {key.replace("_l0", ""): value for key, value in model.state_dict().items()}
    path = tmp_path / "grucell.pt"
    torch.save({"spec": model.spec(), "model": old_names}, path)

    restored, _ = ppo.load_model(path)
    obs = torch.randn(3, toy.OBS_DIM)
    hidden = model.initial_hidden(3)
    assert torch.equal(model.advance(obs, hidden), restored.advance(obs, hidden))


def test_worker_rollouts_are_reproducible():
    """Fresh workers, same weights and seed: the same rollout, bit for bit."""
    pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import ppo

    ppo.seed_everything(3)
    model = ppo.ActorCritic(toy.OBS_DIM, toy.ACTION_NVEC, hidden=8)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    def fresh_rollout(seed):
        with ProcVecEnv(toy.make_line_env, 2) as env:
            env.sync(model.spec(), state, seed)
            return env.rollout(40)

    first, again = fresh_rollout(123), fresh_rollout(123)
    np.testing.assert_array_equal(first["act"], again["act"])
    np.testing.assert_array_equal(first["logp"], again["logp"])
    np.testing.assert_array_equal(first["rew"], again["rew"])
    np.testing.assert_array_equal(first["hid"], again["hid"])
    other = fresh_rollout(124)
    assert not np.array_equal(first["act"], other["act"])


def test_per_type_heads_are_selected_by_the_observation_one_hot():
    """One shared trunk, one head per type (#48): the type picks the head."""
    torch = pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import monster, monster_env, ppo

    ppo.seed_everything(0)
    model = ppo.ActorCritic(
        monster.MONSTER_OBS_DIM,
        monster_env.MONSTER_ACTION_NVEC,
        hidden=8,
        n_types=monster.N_MONSTER_TYPES,
    )
    assert len(model.policy_heads) == monster.N_MONSTER_TYPES
    with torch.no_grad():
        model.policy_heads[1].weight.add_(1.0)  # make the two heads differ

    obs = torch.zeros(2, monster.MONSTER_OBS_DIM)
    obs[0, monster.MONSTER_OBS_BASE_DIM + 0] = 1.0  # Grunt
    obs[1, monster.MONSTER_OBS_BASE_DIM + 1] = 1.0  # Dog
    hidden = torch.ones(2, model.hidden)
    logits = model.head_logits(hidden, obs)
    assert torch.allclose(logits[0], model.policy_heads[0](hidden)[0])
    assert torch.allclose(logits[1], model.policy_heads[1](hidden)[1])
    assert not torch.allclose(logits[0], logits[1])
    # The value is one shared head whatever the body.
    assert torch.equal(model.value(hidden)[0], model.value(hidden)[1])
    with pytest.raises(ValueError, match="observation"):
        model.head_logits(hidden)


def test_warm_start_reproduces_the_single_type_policy():
    """The Grunt's brain carries over exactly; the new type columns are zero."""
    torch = pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import monster, monster_env, ppo

    ppo.seed_everything(5)
    source = ppo.ActorCritic(
        monster.MONSTER_OBS_BASE_DIM, monster_env.MONSTER_ACTION_NVEC, hidden=16
    )
    model = ppo.ActorCritic(
        monster.MONSTER_OBS_DIM,
        monster_env.MONSTER_ACTION_NVEC,
        hidden=16,
        n_types=monster.N_MONSTER_TYPES,
    )
    model.init_from(source)

    # The trunk took the source's input columns and zeroed the new ones.
    assert torch.equal(
        model.gru.weight_ih_l0[:, : monster.MONSTER_OBS_BASE_DIM],
        source.gru.weight_ih_l0,
    )
    assert torch.count_nonzero(model.gru.weight_ih_l0[:, monster.MONSTER_OBS_BASE_DIM :]) == 0
    assert torch.equal(model.gru.weight_hh_l0, source.gru.weight_hh_l0)

    rng = np.random.default_rng(0)
    base = rng.standard_normal((4, monster.MONSTER_OBS_BASE_DIM)).astype(np.float32)
    for index in range(monster.N_MONSTER_TYPES):
        obs = np.zeros((4, monster.MONSTER_OBS_DIM), dtype=np.float32)
        obs[:, : monster.MONSTER_OBS_BASE_DIM] = base
        obs[:, monster.MONSTER_OBS_BASE_DIM + index] = 1.0
        want = source.act(torch.as_tensor(base), source.initial_hidden(4), greedy=True)
        got = model.act(torch.as_tensor(obs), model.initial_hidden(4), greedy=True)
        assert torch.equal(want[0], got[0])  # same actions
        assert torch.allclose(want[1], got[1])  # same log-probability
        assert torch.allclose(want[2], got[2])  # same value
        assert torch.allclose(want[3], got[3])  # same next recurrent state


def test_a_shared_model_checkpoint_round_trips(tmp_path):
    torch = pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import monster, monster_env, ppo

    ppo.seed_everything(6)
    model = ppo.ActorCritic(
        monster.MONSTER_OBS_DIM,
        monster_env.MONSTER_ACTION_NVEC,
        hidden=8,
        n_types=monster.N_MONSTER_TYPES,
    )
    path = tmp_path / "shared.pt"
    ppo.save_checkpoint(path, model, None, {"total_steps": 1})
    restored, saved = ppo.load_model(path)
    assert saved["spec"]["n_types"] == monster.N_MONSTER_TYPES
    obs = torch.randn(3, monster.MONSTER_OBS_DIM)
    obs[:, monster.MONSTER_OBS_BASE_DIM :] = 0.0
    obs[:, monster.MONSTER_OBS_BASE_DIM] = 1.0
    hidden = model.initial_hidden(3)
    assert torch.equal(
        model.act(obs, hidden, greedy=True)[0], restored.act(obs, hidden, greedy=True)[0]
    )


def test_trained_policy_drives_a_shared_model(tmp_path):
    """Evaluation's policy wrapper feeds the type one-hot through (#48)."""
    pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import monster, monster_env, ppo
    from deepquake.eval_nav import TrainedPolicy

    ppo.seed_everything(7)
    model = ppo.ActorCritic(
        monster.MONSTER_OBS_DIM,
        monster_env.MONSTER_ACTION_NVEC,
        hidden=8,
        n_types=monster.N_MONSTER_TYPES,
    )
    path = tmp_path / "shared.pt"
    ppo.save_checkpoint(path, model, None, {"total_steps": 0})
    policy = TrainedPolicy(path, seed=0)
    for index in range(monster.N_MONSTER_TYPES):
        obs = np.zeros(monster.MONSTER_OBS_DIM, dtype=np.float32)
        obs[monster.MONSTER_OBS_BASE_DIM + index] = 1.0
        action = policy(obs)
        assert action.shape == (len(monster_env.MONSTER_ACTION_NVEC),)
        assert all(
            0 <= int(a) < n for a, n in zip(action, monster_env.MONSTER_ACTION_NVEC, strict=True)
        )


def test_monster_list_covers_every_type_in_one_run():
    """A shared-policy run names its classes; unknown ones fail loudly (#48)."""
    from deepquake import train_monster

    assert train_monster.monster_list({"monsters": ["monster_army", "monster_dog"]}) == [
        "monster_army",
        "monster_dog",
    ]
    assert train_monster.monster_list({"monster": "monster_dog"}) == ["monster_dog"]
    assert train_monster.monster_list({}) == ["monster_army"]
    with pytest.raises(ValueError, match="unknown monster classes"):
        train_monster.monster_list({"monsters": ["monster_shambler"]})


def test_the_trainer_warm_starts_and_refuses_a_missing_source(tmp_path):
    torch = pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import monster, monster_env, ppo, train_monster

    ppo.seed_everything(8)
    source = ppo.ActorCritic(
        monster.MONSTER_OBS_BASE_DIM, monster_env.MONSTER_ACTION_NVEC, hidden=8
    )
    path = tmp_path / "grunt.pt"
    ppo.save_checkpoint(path, source, None, {"total_steps": 10})
    model = train_monster.build_model(
        {"init_from": str(path)}, ppo.PPOConfig(hidden=8), torch.device("cpu")
    )
    assert model.n_types == monster.N_MONSTER_TYPES
    base = np.random.default_rng(0).standard_normal((3, monster.MONSTER_OBS_BASE_DIM))
    obs = np.zeros((3, monster.MONSTER_OBS_DIM), dtype=np.float32)
    obs[:, : monster.MONSTER_OBS_BASE_DIM] = base
    obs[:, monster.MONSTER_OBS_BASE_DIM] = 1.0
    want = source.act(
        torch.as_tensor(base, dtype=torch.float32), source.initial_hidden(3), greedy=True
    )
    got = model.act(torch.as_tensor(obs), model.initial_hidden(3), greedy=True)
    assert torch.equal(want[0], got[0])
    with pytest.raises(SystemExit, match="init_from"):
        train_monster.build_model(
            {"init_from": str(tmp_path / "absent.pt")}, ppo.PPOConfig(hidden=8), "cpu"
        )


def test_a_shared_model_warm_starts_into_a_wider_roster(tmp_path):
    """A two-type checkpoint carries into the three-type roster (#50).

    The roster run warm starts from the Grunt's single-type model; the
    three-type run warm starts from the two-type roster model, so the
    Grunt's and the Dog's heads carry over and the new type's column is
    zeroed.
    """
    torch = pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import monster, monster_env, ppo, train_monster

    ppo.seed_everything(9)
    source = ppo.ActorCritic(
        monster.MONSTER_OBS_DIM - 1,
        monster_env.MONSTER_ACTION_NVEC,
        hidden=8,
        n_types=monster.N_MONSTER_TYPES - 1,
    )
    path = tmp_path / "roster.pt"
    ppo.save_checkpoint(
        path,
        source,
        None,
        {"total_steps": 10, "monsters": ["monster_army", "monster_dog"]},
    )
    model = train_monster.build_model(
        {"init_from": str(path)}, ppo.PPOConfig(hidden=8), torch.device("cpu")
    )
    assert model.n_types == monster.N_MONSTER_TYPES

    # The trunk took the source's input columns and zeroed the new one.
    assert torch.equal(model.gru.weight_ih_l0[:, : source.obs_dim], source.gru.weight_ih_l0)
    assert torch.count_nonzero(model.gru.weight_ih_l0[:, source.obs_dim :]) == 0

    # Each old type keeps its own head; the new type starts from the last
    # source head, which is init_from's rule.
    source_heads = source.head_list()
    heads = model.head_list()
    assert len(heads) == monster.N_MONSTER_TYPES
    for index in range(len(source_heads)):
        assert torch.equal(heads[index].weight, source_heads[index].weight)
    assert torch.equal(heads[-1].weight, source_heads[-1].weight)


def test_the_memory_persists_across_rollouts():
    """A worker's recurrent state carries from one rollout into the next."""
    pytest.importorskip("torch", reason="needs the 'train' extra: uv sync --extra train")
    from deepquake import ppo

    ppo.seed_everything(4)
    model = ppo.ActorCritic(toy.OBS_DIM, toy.ACTION_NVEC, hidden=8)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    with ProcVecEnv(toy.make_line_env, 2) as env:
        env.sync(model.spec(), state, 5)
        first = env.rollout(40)
        second = env.rollout(40)
        # The second rollout's first decision is made from the state the
        # first rollout ended in (unless an episode end zeroed it).
        kept = (1.0 - first["done"][-1])[:, None]
        np.testing.assert_allclose(second["hid"][0], first["next_hidden"] * kept)

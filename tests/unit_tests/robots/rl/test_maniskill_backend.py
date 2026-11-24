# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Regressions for native ManiSkill action, camera and evaluation contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from robots.rl import maniskill
from rpent.session import EnvState


class NativeEpisode:
    """Unbatched interface returned by ManiSkill's CPUGymWrapper."""

    def __init__(self, obs, info):
        self.obs = obs
        self.info = info
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        controller = SimpleNamespace(
            joints=[SimpleNamespace(name="joint0"), SimpleNamespace(name="joint1")],
            config=SimpleNamespace(normalize_action=True, use_delta=True),
        )
        self.unwrapped = SimpleNamespace(
            agent=SimpleNamespace(
                uid="native_arm",
                controller=controller,
                single_action_space=self.action_space,
                control_mode="pd_joint_delta_pos",
            )
        )
        self.received = []

    def reset(self, *, seed):
        assert seed == 5
        return self.obs, self.info

    def step(self, action):
        self.received.append(action)
        return (
            self.obs,
            0.5,
            False,
            True,
            {
                **self.info,
                "episode": {"return": 0.5, "episode_len": 1},
            },
        )

    def close(self):
        pass


def make_backend(tmp_path, monkeypatch, obs, info):
    env = NativeEpisode(obs, info)
    monkeypatch.setattr(maniskill, "_make_env", lambda *args: (env, "Native-v1"))
    state = EnvState(tmp_path)
    backend = maniskill.ManiSkillPrimitives(
        task_name="Native-v1", seed=5, state=state, check_cancelled=lambda: None
    )
    return env, backend, state


def test_camera_names_modalities_and_cv_backprojection(tmp_path, monkeypatch):
    rgb = np.full((2, 3, 3), 92, dtype=np.uint8)
    obs = {
        "agent": {"qpos": np.array([0.1, 0.5])},
        "sensor_data": {
            "head": {"rgb": rgb},
            "side_cam": {"rgb": rgb, "depth": np.array([[[0], [1000], [2000]]] * 2)},
        },
        "sensor_param": {
            "side_cam": {
                "extrinsic_cv": np.array([[1, 0, 0, -2], [0, 1, 0, 0], [0, 0, 1, 0]]),
                "intrinsic_cv": np.eye(3),
            }
        },
    }
    _, backend, state = make_backend(tmp_path, monkeypatch, obs, {"success": False})
    assert backend.cameras == ("head", "side_cam")
    assert backend.depth_cameras == ("side_cam",)
    record = backend.record_observation()
    assert record.state == {"agent": {"qpos": [0.1, 0.5]}}
    assert record.extras["success"] is False and record.extras["fail"] is None
    assert "head.png" in record.artifacts and "head_world.npz" not in record.artifacts
    world = state.load("side_cam_world.npz", step=record.step_idx)
    np.testing.assert_allclose(world[0, 1], [3.0, 0.0, 1.0])
    assert np.isnan(world[0, 0]).all()


def test_camera_free_task_without_success_preserves_return_and_truncation(
    tmp_path, monkeypatch
):
    env, backend, _ = make_backend(
        tmp_path, monkeypatch, {"agent": {}, "sensor_data": {}}, {}
    )
    assert not backend.cameras and backend.native_success() is None
    assert backend.action_spec["shape"] == [2]
    backend.record_observation()
    for invalid in ([1, 2], [0], [1, np.nan], [[0, 0]], [True, False]):
        with pytest.raises(ValueError):
            backend.step_action(invalid)
    assert not env.received
    backend.step_action([0.25, -1])
    np.testing.assert_equal(env.received[0], [0.25, -1])
    record = backend.record_observation()
    assert record.terminated is False and record.truncated is True
    assert record.extras["success"] is None and record.extras["episode_return"] == 0.5
    assert record.extras["env_steps"] == 1
    with pytest.raises(RuntimeError, match="ended"):
        backend.step_action([0, 0])


def test_dual_controller_descriptions_follow_native_space_order():
    def arm(name, width):
        space = spaces.Box(-1, 1, shape=(width,), dtype=np.float32)
        return SimpleNamespace(
            uid=name,
            single_action_space=space,
            controller=SimpleNamespace(config=SimpleNamespace(), joints=[]),
        )

    right, left = arm("right", 3), arm("left", 2)
    agent = SimpleNamespace(
        agents_dict={"right": right, "left": left},
        control_mode={"left": "pd_joint_pos", "right": "pd_joint_pos"},
    )
    space = spaces.Dict(
        {"right": right.single_action_space, "left": left.single_action_space}
    )
    env = SimpleNamespace(
        action_space=spaces.utils.flatten_space(space),
        unwrapped=SimpleNamespace(agent=agent, single_action_space=space),
    )
    parts = maniskill._action_description(env)["components"]
    assert [(p["robot"], p["slice"]) for p in parts] == [
        ("left", [0, 2]),
        ("right", [2, 5]),
    ]

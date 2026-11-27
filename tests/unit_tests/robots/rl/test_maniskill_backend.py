# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.

from types import SimpleNamespace

import numpy as np
import pytest

from robots.rl import maniskill
from robots.rl.maniskill_arm import ManiSkillArm, UnsupportedRobotPolicy
from robots.rl.tools import Measurement
from rpent.session import EnvState


class ArmDouble(ManiSkillArm):
    """Keep the shared servo and stub the simulator-specific actuator conversion."""

    def __init__(self, episode, agent):
        self.episode, self.agent = episode, agent
        self._config, self.mount = episode._config, agent.mount
        self.has_gripper, self.tool_rotation = agent.has_gripper, np.eye(3)
        self.opening_target, self.last = 1.0, agent.state.copy()

    def _measure_native(self):
        return Measurement(self.agent.state[:3].copy(), np.eye(3), self.agent.state[3])

    def _pose_action(self, current, xyz, rotation):
        delta, _, distance, angle = self._bounded_pose_step(current, xyz, rotation)
        return np.r_[current.xyz + delta, self.opening_target], distance, angle

    def _command_gripper(self, action, opening, error, within):
        self.opening_target = action[3] = opening


class NativeEpisode:
    instruction = ""

    def __init__(self, obs, info, *, horizon=2, truncate=False):
        pose = np.eye(4)
        pose[0, 3] = 1
        mount = SimpleNamespace(pose=SimpleNamespace(to_transformation_matrix=lambda: pose))
        agent = SimpleNamespace(
            uid="example_arm", mount=mount, has_gripper=True, state=np.array([1, 0, 0.3, 1.0])
        )
        self.unwrapped = SimpleNamespace(agent=agent)
        self.obs, self.info = obs, info
        self.horizon, self.truncate = horizon, truncate
        self.received = []
        self.closed = False

    def reset(self, *, seed):
        self.seed = seed
        return self.obs, self.info

    def step(self, action):
        self.received.append(action)
        self.unwrapped.agent.state[:] = action[0]
        count = len(self.received)
        end = count == self.horizon
        info = {**self.info, "episode": {"return": 2.0 * count, "episode_len": count}}
        return self.obs, 2.0, end and not self.truncate, end and self.truncate, info

    def close(self):
        self.closed = True


def stub_environment(monkeypatch, env, task):
    monkeypatch.setattr(maniskill, "_make_env", lambda *args: (env, task))
    monkeypatch.setattr(maniskill, "ManiSkillArm", ArmDouble)


def test_recorded_camera_calibration_and_video_frames(tmp_path, monkeypatch):
    rgb = np.full((2, 3, 3), 92, dtype=np.uint8)
    depth = np.array([[0, 1000, 2000]] * 2)
    extrinsic = np.eye(4)[:3]
    extrinsic[0, 3] = -2
    obs = {
        "sensor_data": {"head": {"rgb": rgb}, "side": {"rgb": rgb, "depth": depth}},
        "sensor_param": {"side": {"extrinsic_cv": extrinsic, "intrinsic_cv": np.eye(3)}},
    }
    env = NativeEpisode(obs, {"success": False})
    stub_environment(monkeypatch, env, "Native-v1")
    state = EnvState(tmp_path)
    backend = maniskill.ManiSkillPrimitives(
        task_name="Native-v1", seed=5, state=state, check_cancelled=lambda: None
    )
    record = backend.record_observation()
    assert env.seed == 5 and backend.depth_cameras == ("side",)
    assert record.extras["success"] is False and record.extras["fail"] is None
    assert "head.png" in record.artifacts and "head_world.npz" not in record.artifacts
    world = state.load("side_world.npz", step=record.step_idx)
    np.testing.assert_allclose(world[0, 1], [3, 0, 1])
    assert np.isnan(world[0, 0]).all()
    rgb[:] = 93
    backend.step_controllers()
    assert [int(frame[0, 0, 0]) for frame in backend.frames] == [92, 93]


def test_unsupported_robot_does_not_get_a_fabricated_arm():
    episode = SimpleNamespace(_config=None, env_id="Native-v1")
    agent = SimpleNamespace(robot=object(), uid="unmapped")
    with pytest.raises(UnsupportedRobotPolicy, match="no mapped arm"):
        ManiSkillArm(episode, agent)

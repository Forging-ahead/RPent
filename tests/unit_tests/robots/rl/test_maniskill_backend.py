# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Regressions for native ManiSkill action, camera and evaluation contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from robots.rl import maniskill
from rpent.session import EnvState


class Pose:
    def __init__(self, xyz):
        self.matrix = np.eye(4)
        self.matrix[:3, 3] = xyz

    @property
    def p(self):
        return self.matrix[:3, 3]

    def to_transformation_matrix(self):
        return self.matrix.copy()


class NativeEpisode:
    """CPU fixture with a measured offset base and an absolute pose/finger controller."""

    instruction = ""

    def __init__(self, obs, info, *, horizon=2, truncate=False, reward=2.0):
        self.obs, self.info = obs, info
        self.horizon, self.truncate, self.reward = horizon, truncate, reward
        self.closed = False
        self.received = []
        root, wrist, tcp, left, right = [
            SimpleNamespace(index=i, pose=Pose(xyz))
            for i, xyz in enumerate(
                ([1, 0, 0], [1, 0, 0.2], [1, 0, 0.3], [1, 0.04, 0.3], [1, -0.04, 0.3])
            )
        ]
        joints = [
            SimpleNamespace(name="wrist", active_index=0, child_link=wrist),
            SimpleNamespace(name="finger", active_index=1, child_link=left),
        ]
        self.qpos = np.array([[0.0, 0.04]])

        def fk(qpos):
            left.pose.p[1], right.pose.p[1] = qpos[1], -qpos[1]

        pin = SimpleNamespace(
            compute_forward_kinematics=fk,
            get_link_pose=lambda i: [root, wrist, tcp, left, right][i].pose,
        )
        robot = SimpleNamespace(
            root=root,
            active_joints=joints,
            get_qpos=lambda: self.qpos,
            get_qlimits=lambda: np.array([[[-1, 1], [0, 0.04]]]),
            create_pinocchio_model=lambda: pin,
        )
        arm = SimpleNamespace(
            ee_link=tcp,
            root_link=root,
            config=SimpleNamespace(
                use_delta=False,
                normalize_action=False,
                frame="root_translation:root_aligned_body_rotation",
            ),
        )
        grip = SimpleNamespace(
            joints=[joints[1]],
            single_action_space=spaces.Box(-1, 1, (1,)),
            _original_single_action_space=spaces.Box(0, 0.04, (1,)),
            config=SimpleNamespace(lower=0.0, upper=0.04, normalize_action=True),
        )
        controller = SimpleNamespace(
            controllers={"arm": arm, "gripper": grip},
            reset=lambda: None,
            action_mapping={"arm": (0, 6), "gripper": (6, 7)},
        )
        self.unwrapped = SimpleNamespace(
            agent=SimpleNamespace(
                uid="example_arm",
                robot=robot,
                arm_joint_names=["wrist"],
                gripper_joint_names=["finger"],
                supported_control_modes=["pd_ee_pose"],
                control_mode="pd_ee_pose",
                set_control_mode=lambda mode: None,
                single_action_space=spaces.Box(-1, 1, (7,)),
                controller=controller,
                tcp=tcp,
                finger1_link=left,
                finger2_link=right,
            )
        )

    def apply(self, action):
        assert action.shape == (1, 7)
        self.received.append(action.copy())
        agent = self.unwrapped.agent
        agent.tcp.pose.p[:] = agent.robot.root.pose.p + action[0, :3]
        agent.tcp.pose.matrix[:3, :3] = Rotation.from_euler(
            "XYZ", action[0, 3:6]
        ).as_matrix()
        self.qpos[0, 1] = (action[0, 6] + 1) * 0.02

    def reset(self, *, seed):
        self.seed = seed
        self.received.clear()
        return self.obs, self.info

    def step(self, action):
        self.apply(action)
        count = len(self.received)
        end = count == self.horizon
        info = {
            **self.info,
            "episode": {"return": self.reward * count, "episode_len": count},
        }
        return (
            self.obs,
            self.reward,
            end and not self.truncate,
            end and self.truncate,
            info,
        )

    def close(self):
        self.closed = True


def make_backend(tmp_path, monkeypatch, env):
    monkeypatch.setattr(maniskill, "_make_env", lambda *args: (env, "Native-v1"))
    state = EnvState(tmp_path)
    backend = maniskill.ManiSkillPrimitives(
        task_name="Native-v1", seed=5, state=state, check_cancelled=lambda: None
    )
    return backend, state


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
    env = NativeEpisode(obs, {"success": False})
    backend, state = make_backend(tmp_path, monkeypatch, env)
    assert env.seed == 5
    assert backend.cameras == ("head", "side_cam")
    assert backend.depth_cameras == ("side_cam",)
    record = backend.record_observation()
    assert record.state["pose"]["xyz"] == [0, 0, 0.3]
    assert record.state["opening"] == 1
    assert record.extras["success"] is False and record.extras["fail"] is None
    assert "head.png" in record.artifacts and "head_world.npz" not in record.artifacts
    world = state.load("side_cam_world.npz", step=record.step_idx)
    np.testing.assert_allclose(world[0, 1], [3.0, 0.0, 1.0])
    assert np.isnan(world[0, 0]).all()
    rgb[:] = 93
    backend.step_controllers()
    backend.record_observation()
    assert len(backend.frames) == 2
    assert np.all(backend.frames[0] == 92)
    assert np.all(backend.frames[1] == 93)


def test_unsupported_robot_has_no_fabricated_arm(tmp_path, monkeypatch):
    env = NativeEpisode({}, {})
    env.unwrapped.agent.arm_joint_names = []
    with pytest.raises(maniskill.UnsupportedRobotPolicy, match="no mapped arm"):
        make_backend(tmp_path, monkeypatch, env)

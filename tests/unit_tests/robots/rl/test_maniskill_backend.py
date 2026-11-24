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

    def __init__(self, obs, info):
        self.obs, self.info = obs, info
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
        assert seed == 5
        return self.obs, self.info

    def step(self, action):
        self.apply(action)
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
    assert record.state["pose"]["xyz"] == [0, 0, 0.3]
    assert record.state["opening"] == 1
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
    backend.record_observation()
    backend.move_to(xyz=[0.05, 0, 0.3])
    np.testing.assert_allclose(env.received[0][0, :3], [0.03, 0, 0.3])
    record = backend.record_observation()
    assert record.terminated is False and record.truncated is True
    assert record.extras["success"] is None and record.extras["episode_return"] == 0.5
    assert record.extras["env_steps"] == 1
    with pytest.raises(RuntimeError, match="ended"):
        backend.step_controllers()


def test_unsupported_robot_has_no_fabricated_arm(tmp_path, monkeypatch):
    env = NativeEpisode({}, {})
    env.unwrapped.agent.arm_joint_names = []
    monkeypatch.setattr(maniskill, "_make_env", lambda *args: (env, "Cartpole-v1"))
    with pytest.raises(maniskill.UnsupportedRobotPolicy, match="no mapped arm"):
        maniskill.ManiSkillPrimitives(
            task_name="Cartpole-v1",
            seed=5,
            state=EnvState(tmp_path),
            check_cancelled=lambda: None,
        )

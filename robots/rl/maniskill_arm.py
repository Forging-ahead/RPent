# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Translate the shared Cartesian servo's commands into ManiSkill controller inputs."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from robots.rl.coordinates import grasp_axes
from robots.rl.tools import CartesianServo, Measurement


def transform(pose):
    """ManiSkill CPU poses have a leading environment dimension."""
    return np.asarray(pose.to_transformation_matrix(), dtype=float).reshape(4, 4)


def _index(link):
    return int(np.asarray(link.index).item())


def _normalize_target(controller, target):
    space = controller._original_single_action_space
    return 2 * (target - space.low) / (space.high - space.low) - 1


class UnsupportedRobotPolicy(ValueError):
    """The registered robot lacks a public RPent action for this embodiment."""


class ManiSkillArm(CartesianServo):
    def __init__(self, episode, agent):
        self.episode = episode
        self.agent = agent
        self._config = episode._config
        self.robot = agent.robot
        arm_names = getattr(agent, "arm_joint_names", ())
        if not arm_names:
            raise UnsupportedRobotPolicy(
                f"{episode.env_id}: {agent.uid} has no mapped arm for move_to"
            )
        modes = agent.supported_control_modes
        mode = next(
            (m for m in ("pd_ee_pose", "pd_joint_pos") if m in modes),
            agent.control_mode,
        )
        agent.set_control_mode(mode)
        agent.controller.reset()
        ctrl = agent.controller
        self.parts = getattr(ctrl, "controllers", {"arm": ctrl})
        self.arm = self.parts["arm"]
        self.slices = {
            key: slice(*indices)
            for key, indices in getattr(
                ctrl, "action_mapping", {"arm": (0, agent.single_action_space.shape[0])}
            ).items()
        }
        self.joints = {j.name: j for j in self.robot.active_joints}
        gripper_names = getattr(agent, "gripper_joint_names", ())
        self.arm_names = [n for n in arm_names if n not in gripper_names]
        self.mount = (
            self.robot.root
            if agent.uid != "fetch"
            else self.joints[self.arm_names[0]].parent_link
        )
        self.ee = getattr(self.arm, "ee_link", None)
        self.arm_slots = self.slices["arm"]
        self.arm_indices = None
        if self.ee is None:
            self.ee = getattr(agent, "tcp", None) or agent.finger1_link
            self.arm_indices = np.asarray(self.arm.active_joint_indices)
            space = self.arm.single_action_space
            self.arm_bounds = (space.low, space.high)
        self.pin = (
            self.robot.create_pinocchio_model()
            if self.arm_indices is not None or gripper_names
            else None
        )
        self.arm_mask = np.array(
            [j.name in self.arm_names for j in self.robot.active_joints]
        )
        self.gripper = self.parts.get("gripper")
        self.has_gripper = bool(gripper_names)
        self.tool_rotation = np.eye(3)
        if self.has_gripper:
            self._calibrate_gripper(gripper_names)
        self.opening_target = self._opening()
        self.last = self._hold_command()

    def _calibrate_gripper(self, names):
        """Compare fingertip separation at both physical limits using native FK."""
        gripper = self.gripper
        fingers = [
            getattr(self.agent, f"finger{i}_tip", None)
            or getattr(self.agent, f"finger{i}_link")
            for i in (1, 2)
        ]
        drive = names[0]
        if gripper is not None:
            controlled = np.asarray(
                getattr(gripper, "control_joint_indices", [0])
            ).reshape(-1)
            if len(controlled) != 1 or gripper.single_action_space.shape != (1,):
                raise UnsupportedRobotPolicy(
                    f"{self.agent.uid} needs independent finger actions"
                )
            drive = gripper.joints[int(controlled[0])].name
        self.finger_index = int(np.asarray(self.joints[drive].active_index).item())
        ends = np.asarray(self.robot.get_qlimits())[0, self.finger_index]
        if gripper is not None:
            space = getattr(
                gripper, "_original_single_action_space", gripper.single_action_space
            )
            ends = np.clip(ends, space.low, space.high)
        qpos = np.asarray(self.robot.get_qpos())[0].copy()
        gaps, axes = [], []
        for target in ends:
            qpos[self.finger_index] = target
            for name, relation in (
                getattr(getattr(gripper, "config", None), "mimic", {}) or {}
            ).items():
                idx = int(np.asarray(self.joints[name].active_index).item())
                qpos[idx] = target * relation.get("multiplier", 1) + relation.get(
                    "offset", 0
                )
            self.pin.compute_forward_kinematics(qpos)
            first, second = (self.pin.get_link_pose(_index(link)).p for link in fingers)
            tcp = self.pin.get_link_pose(_index(self.ee)).to_transformation_matrix()
            wrist = self.pin.get_link_pose(
                _index(self.joints[self.arm_names[-1]].child_link)
            ).p
            jaw = first - second
            gaps.append(np.linalg.norm(jaw))
            approach = (first + second) / 2 - wrist
            axes.append((tcp[:3, :3].T @ jaw, tcp[:3, :3].T @ approach))
        if ends[0] >= ends[1] or abs(gaps[1] - gaps[0]) < 1e-5:
            raise UnsupportedRobotPolicy(
                f"{self.agent.uid}: cannot calibrate parallel jaw travel"
            )
        opened = int(np.argmax(gaps))
        self.closed_position, self.open_position = ends[1 - opened], ends[opened]
        jaw, approach = axes[opened]
        jaw = jaw / np.linalg.norm(jaw)
        approach = approach - jaw * (jaw @ approach)
        if self.agent.uid in {"panda", "panda_wristcam"}:
            # Use the robot's grasp convention at its designated hand TCP.
            grasp = self.agent.build_grasp_pose(np.eye(3)[2], np.eye(3)[0], np.zeros(3))
            self.tool_rotation = transform(grasp)[:3, :3].T
        else:
            self.tool_rotation = grasp_axes(jaw, approach)

    def _opening(self):
        if not self.has_gripper:
            return 0.0
        q = np.asarray(self.robot.get_qpos())[0, self.finger_index]
        travel = self.open_position - self.closed_position
        return float(np.clip((q - self.closed_position) / travel, 0, 1))

    def _tcp(self):
        pose = getattr(self.agent, "tcp_pose", None)
        return transform(self.ee.pose if pose is None else pose)

    def _measure_native(self):
        tcp = self._tcp()
        return Measurement(tcp[:3, 3], tcp[:3, :3], self._opening())

    @property
    def coordinates(self):
        return self.episode.coordinates

    @property
    def ended(self):
        return self.episode.ended

    def _ended_reason(self):
        return "terminated" if self.episode.terminated else "truncated"

    def _hold_command(self):
        command = np.zeros(self.agent.single_action_space.shape, dtype=np.float32)
        for name, component in self.parts.items():
            if (
                name not in {"arm", "gripper", "base"}
                and not component.config.use_delta
            ):
                command[self.slices[name]] = np.asarray(component.qpos)[0]
        if self.arm_indices is not None:
            command[self.arm_slots] = np.asarray(self.arm.qpos)[0]
        else:
            command[self.arm_slots] = self._cartesian_input(transform(self.ee.pose))
        if self.has_gripper:
            self._command_gripper(command, self.opening_target, 0, True)
        return command

    def _cartesian_input(self, world):
        ctrl = self.arm
        relative = np.linalg.solve(transform(ctrl.root_link.pose), world)
        if ctrl.config.use_delta:
            # The Bridge controller integrates deltas from its previous target.
            reference = transform(
                ctrl._target_pose if ctrl.config.use_target else ctrl.ee_pose_at_base
            )
            relative[:3, 3] -= reference[:3, 3]
            relative[:3, :3] = relative[:3, :3] @ reference[:3, :3].T
        if (
            ctrl.config.frame != "root_translation:root_aligned_body_rotation"
            or ctrl.config.normalize_action
        ):
            raise UnsupportedRobotPolicy(
                f"Unsupported native Cartesian convention for {self.agent.uid}"
            )
        return np.r_[
            relative[:3, 3], Rotation.from_matrix(relative[:3, :3]).as_euler("XYZ")
        ]

    def _pose_action(self, current, xyz, rotation):
        position, rotation_step, distance, angle = self._bounded_pose_step(
            current, xyz, rotation
        )
        target = np.eye(4)
        target[:3, 3] = current.xyz + position
        target[:3, :3] = (
            Rotation.from_rotvec(rotation_step).as_matrix() @ current.rotation
        )
        # SO100's grasp centre follows the moving jaw. Re-measure the link offset each step.
        link_target = target @ np.linalg.solve(self._tcp(), transform(self.ee.pose))
        command = self.last.copy()
        if self.arm_indices is None:
            command[self.arm_slots] = self._cartesian_input(link_target)
        else:
            command[self.arm_slots] = self._joint_target(link_target)
        if self.has_gripper:
            self._command_gripper(command, self.opening_target, 0, True)
        return command, distance, angle

    def _joint_target(self, link_target):
        import sapien

        goal = np.linalg.solve(transform(self.robot.root.pose), link_target)
        solution, _, _ = self.pin.compute_inverse_kinematics(
            _index(self.ee),
            sapien.Pose(goal),
            initial_qpos=np.asarray(self.robot.get_qpos())[0],
            active_qmask=self.arm_mask,
            max_iterations=100,
        )
        return np.clip(solution[self.arm_indices], *self.arm_bounds)

    def _command_gripper(self, action, opening, error, within):
        self.opening_target = opening
        q = self.closed_position + opening * (self.open_position - self.closed_position)
        if self.gripper is None:
            # SO100 uses one combined joint-position controller.
            action[self.finger_index] = q
        else:
            ctrl = self.gripper
            if ctrl.config.normalize_action:
                q = _normalize_target(ctrl, q)
            action[self.slices["gripper"]] = q

    def _step(self, action):
        # G1's two arms share the same body target vector.
        self.last[:] = action
        self.episode.step_controllers()


class ManiSkillJointArm(ManiSkillArm):
    """Use selected channels of an absolute joint-position controller."""

    def __init__(
        self, episode, agent, tcp, mount, targets, channels, arm, fingers, axes
    ):
        self.episode = episode
        self._config = episode._config
        self.robot = agent.robot
        self.ee, self.mount = tcp, mount
        self.last = targets
        joints = self.robot.active_joints_map
        self.arm_slots = [channels[n] for n in arm]
        self.arm_indices = [int(np.asarray(joints[n].active_index).item()) for n in arm]
        self.arm_mask = np.isin(np.arange(self.robot.dof[0]), self.arm_indices)
        self.pin = self.robot.create_pinocchio_model()
        limits = np.asarray(self.robot.get_qlimits())[0]
        self.arm_bounds = limits[self.arm_indices].T
        self.has_gripper = True
        self.finger_slots = [channels[n] for n in fingers]
        self.finger_indices = [
            int(np.asarray(joints[n].active_index).item()) for n in fingers
        ]
        limits = limits[self.finger_indices]
        # Both robots use zero as their open pose. Flex toward the longer travel.
        self.closed = (
            0.7 * limits[np.arange(len(fingers)), np.abs(limits).argmax(axis=1)]
        )
        self.tool_rotation = grasp_axes(*axes)
        self.opening_target = self._opening()

    def _tcp(self):
        return transform(self.ee.pose)

    def _opening(self):
        qpos = np.asarray(self.robot.get_qpos())[0, self.finger_indices]
        return float(
            np.clip(1 - qpos @ self.closed / (self.closed @ self.closed), 0, 1)
        )

    def _command_gripper(self, action, opening, error, within):
        self.opening_target = opening
        action[self.finger_slots] = (1 - opening) * self.closed


def _joint_position_channels(agent):
    """Measure initial hold targets in the native action order, including mimic hands."""
    agent.set_control_mode("pd_joint_pos")
    agent.controller.reset()
    controller = agent.controller
    targets = np.zeros(agent.single_action_space.shape, dtype=np.float32)
    channels = {}
    for name, part in controller.controllers.items():
        start, end = controller.action_mapping[name]
        if start == end:  # Passive joints have state, but no action channels.
            continue
        indices = getattr(part, "control_joint_indices", range(len(part.joints)))
        targets[start:end] = np.asarray(part.qpos)[0, indices]
        channels.update(
            (part.joints[int(j)].name, start + i) for i, j in enumerate(indices)
        )
    return targets, channels


def g1_arms(episode, agent):
    targets, channels = _joint_position_channels(agent)
    arms = {}
    for side in ("left", "right"):
        arm = [
            name
            for name in channels
            if name.startswith((f"{side}_shoulder", f"{side}_elbow"))
        ]
        arms[side] = ManiSkillJointArm(
            episode,
            agent,
            getattr(agent, f"{side}_tcp"),
            agent.robot.active_joints_map[arm[0]].child_link,
            targets,
            channels,
            arm,
            getattr(agent, f"{side}_finger_joints"),
            ([0, 1, 0], [1, 0, 0]),
        )
    return arms


def floating_hand(episode, agent):
    targets, channels = _joint_position_channels(agent)
    return ManiSkillJointArm(
        episode,
        agent,
        agent.robot.links_map[agent.ee_link_name],
        agent.robot.root,
        targets,
        channels,
        agent.root_joint_names,
        [n for n in channels if n not in agent.root_joint_names],
        ([1, 0, 0], [0, 0, 1]),
    )

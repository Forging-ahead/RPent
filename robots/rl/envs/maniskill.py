# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native ManiSkill manipulation episodes for the shared simulator pool."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from robots.rl.envs.libero import (
    ControlConfig,
    LiberoRLPrimitives,
    Measurement,
    _clip_norm,
)
from robots.rl.envs.robocasa import RoboCasaRLPrimitives
from robots.rl.geometry import Coordinates, grasp_axes, world_from_depth
from robots.rl.perception import RecordedPerception
from robots.rl.toolkit import ActionBatch, DualArmBatch, MobileBatch, MoveAction
from rpent.tools import ToolResult, tool


class ToolTipBatch(ActionBatch):
    @tool
    def execute_actions(
        self, actions: Annotated[list[MoveAction], Field(min_length=1)]
    ) -> ToolResult:
        """Execute 1..K move_to items for the tool tip. Each item is measured before continuing."""
        return self._execute_actions(actions)


def _task_hint(env_id, image_size=512):
    instructions = (
        Path(__file__).resolve().parents[1] / "prompts" / "maniskill_tasks.md"
    ).read_text(encoding="utf-8")
    section = instructions.partition(f"\n## {env_id}\n")[2].partition("\n## ")[0]
    hint = " ".join(section.split())
    if image_size != 512 and "512px" in hint:
        hint = (
            "For the pixel formulas below, multiply displayed rows, columns and box "
            f"edges by 512/{image_size} first. Use displayed pixels for tool calls. "
        ) + hint
    return hint


def _make_env(
    task_name: str,
    render_device: int | None,
    max_episode_steps: int | None,
    seed: int = 0,
):
    import mani_skill.envs  # noqa: F401 - imports native task registrations
    from mani_skill.sensors.camera import parse_camera_configs
    from mani_skill.utils import common
    from mani_skill.utils.registration import REGISTERED_ENVS
    from omegaconf import OmegaConf
    from rlinf.agents.rpent.maniskill_tasks import CARTESIAN_TASKS
    from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv
    from rlinf.envs.sim.maniskill.peg_insertion_side_variants import (
        resolve_maniskill_task_descriptions,
    )

    env_id = "PickCube-v1" if task_name == "pick_cube" else task_name
    if env_id not in CARTESIAN_TASKS:
        raise ValueError(f"Unsupported ManiSkill pose task: {env_id}")
    if env_id not in REGISTERED_ENVS:
        raise ValueError(f"Unknown ManiSkill task {env_id!r}")
    spec = REGISTERED_ENVS[env_id]
    modes = spec.cls.SUPPORTED_OBS_MODES
    if "any_textures" in modes or "rgbd" in modes:
        mode = "rgbd"
    else:
        mode = next((m for m in modes if "rgb" in m), modes[0])
    horizon = spec.max_episode_steps
    if max_episode_steps is not None:
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        horizon = min(horizon, max_episode_steps) if horizon else max_episode_steps

    class RPentManiskillEnv(ManiskillEnv):
        @property
        def instruction(self):
            return resolve_maniskill_task_descriptions(
                self.unwrapped, num_envs=1, is_peg_insertion_side=False
            )[0]

        def _wrap_obs(self, raw_obs, infos=None):
            # RPent records individual camera frames and their calibration.
            return common.unbatch(common.to_numpy(raw_obs))

        @property
        def unwrapped(self):
            return self.env.unwrapped

        def close(self):
            self.env.close()

    cfg = OmegaConf.create(
        {
            "seed": seed,
            "group_size": 1,
            "auto_reset": False,
            "use_fixed_reset_state_ids": False,
            "ignore_terminations": False,
            "reward_mode": "raw",
            "use_rel_reward": False,
            "video_cfg": {"info_on_video": False},
            "init_params": {
                "id": env_id,
                "obs_mode": mode,
                "sim_backend": "physx_cpu",
                "render_backend": "gpu"
                if render_device is None
                else f"cuda:{render_device}",
                "max_episode_steps": horizon,
                "sensor_configs": {"width": 512, "height": 512},
                "human_render_camera_configs": {"width": 512, "height": 512},
            },
        }
    )
    try:
        env = RPentManiskillEnv(
            cfg, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None
        )
    except (EOFError, SystemExit) as e:
        raise RuntimeError(
            f"ManiSkill could not initialize {env_id}; check its required assets"
        ) from e
    native = env.unwrapped
    # Bridge's calibrated camera and real-image overlay must cover the same rays.
    # FoV cameras derive their intrinsics directly from the actual render size.
    if any(c.intrinsic is not None for c in native._sensor_configs.values()):
        try:
            defaults = parse_camera_configs(native._default_sensor_configs)
            defaults.update(parse_camera_configs(native.agent._sensor_configs))
            for name, camera in defaults.items():
                if camera.intrinsic is not None:
                    scale = np.diag([512 / camera.width, 512 / camera.height, 1])
                    native._custom_sensor_configs[name] = {
                        "intrinsic": scale @ camera.intrinsic
                    }
            env.reset(seed=seed, options={"reconfigure": True})
        except BaseException:
            env.close()
            raise
    return env, env_id


def horizontal_rotation(forward: np.ndarray) -> np.ndarray:
    """Return forward, left and up axes from a measured heading in world space."""
    x, y = forward[:2]
    length = np.linalg.norm([x, y, 0.0])
    if length < 1e-6:
        raise ValueError("robot base forward axis must have a horizontal projection")
    x, y = x / length, y / length
    return np.array([[x, -y, 0], [y, x, 0], [0, 0, 1]])


def dual_arm_mount(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the common arm origin and heading from both mounting centres."""
    return (left + right) / 2, horizontal_rotation(np.cross(left - right, [0, 0, 1]))


class LocalManiSkillEnv:
    """Own the seeded native RLinf environment for one simulator-pool session."""

    def __init__(
        self,
        task_name: str,
        seed: int,
        max_episode_steps: int = 10000,
        render_device: int | None = None,
        image_size: int = 512,
    ):
        self.scene, self.env_id = _make_env(
            task_name, render_device, max_episode_steps, seed
        )
        try:
            self.obs, self.info = self.scene.reset(seed=seed)
            self.last_info = {"init_state_id": seed}
            self.task_language = self.scene.instruction
            if not self.task_language:
                doc = inspect.cleandoc(type(self.scene.unwrapped).__doc__ or "")
                description = (
                    doc.removeprefix("**Task Description:**").strip().split("\n\n")[0]
                )
                self.task_language = (
                    f"{self.env_id}: {description}" if description else self.env_id
                )
            hint = _task_hint(self.env_id, image_size)
            if hint:
                self.task_language += " " + hint.replace(
                    "{max_episode_steps}",
                    str(self.scene.cfg.init_params.max_episode_steps),
                )
        except BaseException:
            self.close()
            raise

    def get_task_language(self) -> str:
        return self.task_language

    def close(self) -> None:
        self.scene.close()


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


class ManiSkillArm(LiberoRLPrimitives):
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

    def _bounded_pose_step(
        self, current: Measurement, xyz: np.ndarray, rotation: np.ndarray
    ):
        """World-frame translation and rotation increments, plus unclipped errors."""
        position = xyz - current.xyz
        angle = Rotation.from_matrix(rotation @ current.rotation.T).as_rotvec()
        return (
            _clip_norm(position, self._config.position_step_m),
            _clip_norm(angle, self._config.rotation_step_rad),
            float(np.linalg.norm(position)),
            float(np.linalg.norm(angle)),
        )

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


class ManiSkillRLPrimitives(RecordedPerception):
    """Apply shared Cartesian and mobile control to the native robot's controllers."""

    use_requested_display_size = True

    @staticmethod
    def tool_profile(toolkit, perception):
        """Select the shared action batch and cameras supported by this robot."""
        if perception is None:
            raise ValueError(
                "ManiSkill tool declarations require the active robot and cameras"
            )
        capabilities = perception.tool_capabilities
        if capabilities.get("mobile"):
            batch = MobileBatch(toolkit).execute_actions
            schema = batch.input_schema
            schema["$defs"]["BaseMove"]["properties"]["vy"].update(
                minimum=0,
                maximum=0,
                description="Fetch differential drive requires zero lateral velocity. Turn before driving sideways.",
            )
            batch = replace(batch, _input_schema=schema)
        elif not capabilities["gripper"]:
            batch = ToolTipBatch(toolkit).execute_actions
        else:
            batch_type = DualArmBatch if capabilities["dual_arm"] else ActionBatch
            batch = batch_type(toolkit).execute_actions
        rgb = capabilities["rgb_cameras"]
        return batch, (
            rgb,
            rgb if capabilities["segment"] else (),
            capabilities["depth_cameras"],
        )

    BASE_LIMITS = RoboCasaRLPrimitives.BASE_LIMITS
    move_base = RoboCasaRLPrimitives.move_base
    move_lift = RoboCasaRLPrimitives.move_lift
    _mobility_delta = RoboCasaRLPrimitives._mobility_delta
    _mobility_feedback = RoboCasaRLPrimitives._mobility_feedback

    def __init__(self, *, env: LocalManiSkillEnv, max_episode_steps: int, **kwargs):
        self._init_perception(
            env=env,
            max_episode_steps=env.scene.cfg.init_params.max_episode_steps,
            **kwargs,
        )
        self._config = ControlConfig()
        self.env_id = env.env_id
        self._obs, self._info = env.obs, env.info
        sensors = self._obs.get("sensor_data", {})
        params = self._obs.get("sensor_param", {})
        self.cameras = tuple(name for name, data in sensors.items() if "rgb" in data)
        self.depth_cameras = tuple(
            name
            for name in self.cameras
            if "depth" in sensors[name]
            and {"intrinsic_cv", "extrinsic_cv"} <= params.get(name, {}).keys()
        )
        self.env_steps = 0
        self.episode_return = 0.0
        self.reward = None
        self.terminated = self.truncated = False
        native = self.env.scene.unwrapped
        group = getattr(native.agent, "agents_dict", None)
        self.agent_keys = None
        self._mount_forward_axis = 0
        if group is not None:
            agents = {
                side: getattr(native, f"{side}_agent", None)
                for side in ("left", "right")
            }
            if len(group) != 2 or any(agent is None for agent in agents.values()):
                raise UnsupportedRobotPolicy(
                    f"{self.env_id} needs a physical arm mapping"
                )
            self.arms = {
                side: ManiSkillArm(self, agent) for side, agent in agents.items()
            }
            self.agent_keys = {
                side: next(key for key, member in group.items() if member is agent)
                for side, agent in agents.items()
            }
        elif native.agent.uid in {
            "unitree_g1_simplified_upper_body",
            "unitree_g1_simplified_upper_body_with_head_camera",
        }:
            self.arms = g1_arms(self, native.agent)
        elif native.agent.uid == "floating_ability_hand_right":
            self.arms = {"arm": floating_hand(self, native.agent)}
            # The floating hand's fixed root has its +X axis pointing down.
            self._mount_forward_axis = 2
        else:
            self.arms = {"arm": ManiSkillArm(self, native.agent)}
        self.dual_arm = len(self.arms) == 2
        self.mobile = not self.dual_arm and native.agent.uid == "fetch"
        if self.mobile:
            arm = self.arms["arm"]
            self._lift_index = [joint.name for joint in arm.parts["body"].joints].index(
                "torso_lift_joint"
            )
            base = arm.parts["base"]._original_single_action_space
            body = arm.parts["body"]._original_single_action_space
            self._meta = {
                "control_freq": native.control_freq,
                "parts": {
                    "base": {
                        "actuator_min": [base.low[0], 0, base.low[1]],
                        "actuator_max": [base.high[0], 0, base.high[1]],
                    },
                    "torso": {
                        "output_min": [body.low[self._lift_index]],
                        "output_max": [body.high[self._lift_index]],
                    },
                },
            }
        self.has_gripper = bool(self.arms) and all(
            arm.has_gripper for arm in self.arms.values()
        )
        self.tool_capabilities = {
            "rgb_cameras": list(self.cameras),
            "depth_cameras": list(self.depth_cameras),
            "segment": bool(self.cameras) and self._sam3 is not None,
            "dual_arm": self.dual_arm,
            "mobile": self.mobile,
            "gripper": self.has_gripper,
        }
        table = getattr(getattr(native, "table_scene", None), "table", None)
        mesh = table.get_first_collision_mesh() if table is not None else None
        self.table_z = None if mesh is None else float(mesh.bounds[1, 2])
        self._refresh_coordinates()
        self._save_artifact("coordinates.json", self.coordinates.metadata(), None)

    @property
    def ended(self) -> bool:
        return self.terminated or self.truncated

    def native_success(self) -> bool:
        episode = self._info.get("episode", {})
        success = episode.get("success_once", self._info.get("success"))
        if success is None:
            raise ValueError(f"{self.env_id} did not report its native success flag")
        return bool(success)

    def native_outcome(self) -> dict:
        fail = self._info.get("episode", {}).get("fail_once", self._info.get("fail"))
        return {
            "success": self.native_success(),
            "fail": None if fail is None else bool(fail),
            "episode_return": self.episode_return,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    def robot_state(self) -> dict:
        states = {}
        for name, arm in self.arms.items():
            measured = arm.measure().state()
            if not arm.has_gripper:
                measured.pop("opening")
            states[name] = measured
        if self.mobile:
            arm = self.arms["arm"]
            states["arm"]["lift_m"] = float(
                np.asarray(arm.parts["body"].qpos)[0, self._lift_index]
            )
        if self.dual_arm:
            return states
        state = states["arm"]
        if self.env_id == "PickCube-v1":
            # The goal marker is intentionally hidden from ManiSkill policy cameras.
            extra = self._obs["extra"]
            goal = self.coordinates.points(np.asarray(extra["goal_pos"]))
            state["goal"] = {"xyz": goal.round(4).tolist()}
            state["is_grasped"] = bool(extra["is_grasped"])
            state["cube_size_m"] = 2 * self.env.scene.unwrapped.cube_half_size
        return state

    def _refresh_coordinates(self):
        mounts = [transform(arm.mount.pose) for arm in self.arms.values()]
        if self.dual_arm:
            origin, rotation = dual_arm_mount(mounts[0][:3, 3], mounts[1][:3, 3])
        else:
            origin = mounts[0][:3, 3].copy()
            rotation = horizontal_rotation(mounts[0][:3, self._mount_forward_axis])
        coords = Coordinates(
            origin,
            rotation,
            table_height_m=None if self.table_z is None else self.table_z - origin[2],
            tcp="native_grasp_tcp" if self.has_gripper else "native_tool_tip",
        )
        self.coordinates = (
            coords.after(self.coordinates) if hasattr(self, "coordinates") else coords
        )

    def move_to(self, *, arm="arm", **target):
        return self.arms[arm].move_to(**target)

    def set_gripper(self, *, opening, arm="arm"):
        return self.arms[arm].set_gripper(opening=opening)

    def _mobility_state(self):
        from scipy.spatial.transform import Rotation

        chassis = transform(self.arms["arm"].agent.base_link.pose)
        return (
            chassis[:3, 3].copy(),
            Rotation.from_matrix(chassis[:3, :3]),
            self.coordinates.origin.copy(),
        )

    def _mobile_step(self, velocity=None, lift=0.0, *, check_cancelled=True):
        arm = self.arms["arm"]
        command = arm.last.copy()
        # Fetch has a differential drive and a delta-position torso in pd_joint_pos mode.
        twist = np.zeros(2) if velocity is None else np.asarray(velocity)[[0, 2]]
        delta = np.zeros(arm.parts["body"]._original_single_action_space.shape)
        delta[self._lift_index] = lift
        for name, target in (("base", twist), ("body", delta)):
            command[arm.slices[name]] = _normalize_target(arm.parts[name], target)
        arm.last = command
        self.step_controllers(check_cancelled=check_cancelled)

    def step_controllers(self, *, check_cancelled=True) -> None:
        if check_cancelled:
            self._check_cancelled()
        if self.ended:
            raise RuntimeError("ManiSkill episode has ended")
        # Explicit batch dimensions remain valid after an official control-mode switch.
        if self.agent_keys is not None:
            action = {
                self.agent_keys[k]: arm.last[None, :] for k, arm in self.arms.items()
            }
        else:
            action = next(iter(self.arms.values())).last[None, :]
        self._obs, reward, terminated, truncated, self._info = self.env.scene.step(
            action
        )
        self.terminated, self.truncated = bool(terminated), bool(truncated)
        self.reward = float(reward)
        episode = self._info["episode"]
        self.episode_return = float(episode["return"])
        self.env_steps = int(episode["episode_len"])

        if self.record_artifacts and self.cameras:
            self.frames.append(self._obs["sensor_data"][self.cameras[0]]["rgb"].copy())
        self._refresh_coordinates()

    def record_observation(self, **kwargs):
        # The control loop already captured the last native frame for this observation.
        has_native_frame = bool(self.frames)
        record = super().record_observation(
            extras={**self.native_outcome(), "reward": self.reward}, **kwargs
        )
        if self.record_artifacts and has_native_frame:
            self.frames.pop()
        return record

    def terminal_flags(self, state: dict) -> dict[str, bool]:
        return {"terminated": self.terminated, "truncated": self.truncated}

    def capture(self, camera: str) -> tuple[np.ndarray, np.ndarray, dict]:
        data = self._obs["sensor_data"][camera]
        rgb = data["rgb"]
        if camera not in self.depth_cameras:
            return rgb, np.full((*rgb.shape[:2], 3), np.nan, dtype=np.float32), {}
        params = self._obs["sensor_param"][camera]
        extrinsic = np.asarray(params["extrinsic_cv"], dtype=float)
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))
        elif extrinsic.shape != (4, 4):
            raise ValueError("camera extrinsic must be a 3x4 or 4x4 transform")
        calibration = {
            "intrinsic_K": np.asarray(params["intrinsic_cv"]).tolist(),
            "extrinsic_cam2world": np.linalg.inv(extrinsic).tolist(),
        }
        # ManiSkill reports axial depth in millimetres, zero marks missing pixels.
        depth = np.asarray(data["depth"], dtype=np.float32) / 1000
        if depth.ndim == 3:
            depth = depth.squeeze(axis=-1)
        depth[depth <= 0] = np.nan
        points = world_from_depth(depth, calibration).astype(np.float32)
        return rgb, points, calibration

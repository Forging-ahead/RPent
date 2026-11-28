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

import itertools
from dataclasses import dataclass, replace
from typing import Annotated, Any

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from robots.rl import geometry
from robots.rl.geometry import (
    Coordinates,
    grasp_axes,
    mounting_calibration,
    world_from_depth,
)
from robots.rl.perception import RecordedPerception
from robots.rl.toolkit import ActionBatch, DualArmBatch, MobileBatch, MoveAction
from robots.rl.units import CAMERAS
from rpent.tools import ToolResult, tool

CATALOG = {
    "tasks": {
        "DrawSVG-v1": "Trace the outline on the canvas.",
        "DrawTriangle-v1": "Trace the triangle on the canvas.",
        "FMBAssembly1Easy-v1": "Assemble the bridge on the board.",
        "InsertFlower-v1": "Insert the flower into the vase.",
        "LiftPegUpright-v1": "Stand the peg upright.",
        "PegInsertionSide-v1": "Insert the peg into the side hole.",
        "PickCube-v1": "Move the cube to the target.",
        "PickCubeSO100-v1": "Move the cube to the target.",
        "PickCubeWidowXAI-v1": "Move the cube to the target.",
        "PickSingleYCB-v1": "Move the object to the target.",
        "PlaceSphere-v1": "Place the sphere in the bin.",
        "PlugCharger-v1": "Plug the charger into the outlet.",
        "PokeCube-v1": "Use the peg to push the cube to the target.",
        "PullCube-v1": "Pull the cube to the target.",
        "PullCubeTool-v1": "Use the tool to pull the cube to the target.",
        "PushCube-v1": "Push the cube to the target.",
        "PushT-v1": "Push the T-shaped block onto its target.",
        "RollBall-v1": "Roll the ball to the target.",
        "SO100GraspCube-v1": "Grasp the cube.",
        "StackCube-v1": "Stack the red cube on the green cube.",
        "StackPyramid-v1": "Stack the cubes into a pyramid with the blue cube on top.",
        "TwoRobotPickCube-v1": "Use both arms to move the cube to the target.",
        "TwoRobotStackCube-v1": "Use both arms to stack the cubes.",
        "UnitreeG1PlaceAppleInBowl-v1": "Place the apple in the bowl.",
        "UnitreeG1TransportBox-v1": "Transport the box to the target.",
    },
    "optional_tasks": {
        "PutCarrotOnPlateInScene-v1": "Put the carrot on the plate.",
        "PutEggplantInBasketScene-v1": "Put the eggplant in the basket.",
        "PutSpoonOnTableClothInScene-v1": "Put the spoon on the tablecloth.",
        "StackGreenCubeOnYellowCubeBakedTexInScene-v1": "Stack the green cube on the yellow cube.",
        "AssemblingKits-v1": "Place the shape into its matching cutout.",
        "OpenCabinetDoor-v1": "Open the cabinet door.",
        "OpenCabinetDrawer-v1": "Open the cabinet drawer.",
        "PickClutterYCB-v1": "Pick the designated object from the clutter.",
        "TurnFaucet-v1": "Turn the faucet handle.",
    },
}
TASK_DESCRIPTIONS = {**CATALOG["tasks"], **CATALOG["optional_tasks"]}


class ToolTipBatch(ActionBatch):
    @tool
    def execute_actions(
        self, actions: Annotated[list[MoveAction], Field(min_length=1)]
    ) -> ToolResult:
        """Execute 1..K ordered move_to commands for the pen tip. Supply xyz, orientation_rpy, or both. Each move uses the measured start pose for omitted fields. Execution stops on success, truncation, error or partial movement. Read executed_count, stop_reason and final_state before continuing."""
        return self._execute_actions(actions)


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
    from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv

    env_id = "PickCube-v1" if task_name == "pick_cube" else task_name
    if env_id not in TASK_DESCRIPTIONS:
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

        except BaseException:
            env.close()
            raise
    return env, env_id


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
            self.obs, self.info = self.scene.reset(
                seed=seed,
                options={"reconfigure": True}
                if any(
                    c.intrinsic is not None
                    for c in self.scene.unwrapped._sensor_configs.values()
                )
                else None,
            )
            self.last_info = {"init_state_id": seed}
            self.task_language = TASK_DESCRIPTIONS[self.env_id]
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


@dataclass(frozen=True)
class ControlConfig:
    """Backend servo limits; planners never set these."""

    position_step_m: float = 0.03
    rotation_step_rad: float = 0.12
    position_tolerance_m: float = 0.004
    rotation_tolerance_rad: float = 0.02
    settle_speed_m: float = 0.001
    settle_speed_rad: float = 0.005
    max_motion_steps: int = 300
    stall_steps: int = 15
    stall_progress_m: float = 0.002
    stall_progress_rad: float = 0.01
    opening_tolerance: float = 0.08
    max_gripper_steps: int = 40
    settle_steps: int = 3
    settle_opening: float = 0.01


@dataclass(frozen=True)
class Measurement:
    """Measured end-effector pose and normalized gripper opening."""

    xyz: np.ndarray
    rotation: np.ndarray
    opening: float

    def state(self) -> dict[str, Any]:
        return {
            "pose": {
                "xyz": [round(float(value), 4) for value in self.xyz],
                "orientation_rpy": geometry.rpy_of(self.rotation),
            },
            "opening": round(self.opening, 3),
        }


def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= limit else vector * (limit / norm)


def _stalled(errors: list[tuple[float, float]], config: ControlConfig) -> bool:
    if len(errors) <= config.stall_steps:
        return False
    (old_d, old_a), (d, a) = errors[-1 - config.stall_steps], errors[-1]
    return old_d - d < config.stall_progress_m and old_a - a < config.stall_progress_rad


class ManiSkillArm:
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
        self.fingertips = fingers
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

    def measure(self) -> Measurement:
        native = self._measure_native()
        return Measurement(
            self.coordinates.points(native.xyz),
            self.coordinates.orientation(native.rotation @ self.tool_rotation),
            native.opening,
        )

    @tool(
        json_schema_extra={
            "additionalProperties": False,
            "anyOf": [
                {"required": ["xyz"], "properties": {"xyz": {"type": "array"}}},
                {
                    "required": ["orientation_rpy"],
                    "properties": {"orientation_rpy": {"type": "array"}},
                },
            ],
        }
    )
    def move_to(
        self,
        xyz: Annotated[list[float], Field(min_length=3, max_length=3)] | None = None,
        orientation_rpy: Annotated[list[float], Field(min_length=3, max_length=3)]
        | None = None,
    ) -> ToolResult:
        """Servo the grip_site toward an absolute world pose, without collision avoidance. Supply xyz, orientation_rpy, or both; an omitted part keeps its measured value at the start of this action. The gripper keeps its previous opening target. Returns reached only within 4mm and 0.02rad; partial means the target was not reached and reports a reason. Use the returned measured pose, never assume the command succeeded. Example: {"xyz":[0.1,0,0.3],"orientation_rpy":[3.14159265,0,0]}.

        Args:
            xyz: Absolute grip-site position [x,y,z] in world metres; three finite numbers.
            orientation_rpy: Absolute grip-site [roll,pitch,yaw] in radians, fixed world-axis XYZ: R=Rz(yaw)@Ry(pitch)@Rx(roll). [pi,0,yaw] points the approach axis down. Omit to keep the complete measured rotation.
        """
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        if xyz is None and orientation_rpy is None:
            return ToolResult(data={"error": "Supply xyz, orientation_rpy, or both."})
        start = self.measure()
        target = self.coordinates.world_position(
            start.xyz if xyz is None else np.asarray(xyz, dtype=np.float64)
        )
        goal = (
            self.coordinates.world_orientation(start.rotation)
            if orientation_rpy is None
            else self.coordinates.world_orientation(
                Rotation.from_euler("xyz", orientation_rpy).as_matrix()
            )
        ) @ self.tool_rotation.T
        config = self._config
        errors: list[tuple[float, float]] = []
        reason = None
        previous = self._measure_native()
        for steps in itertools.count():
            current = self._measure_native()
            action, distance, angle = self._pose_action(current, target, goal)
            errors.append((distance, angle))
            settled = (
                np.linalg.norm(current.xyz - previous.xyz) < config.settle_speed_m
                and Rotation.from_matrix(
                    current.rotation @ previous.rotation.T
                ).magnitude()
                < config.settle_speed_rad
            )
            previous = current
            within = (
                distance < config.position_tolerance_m
                and angle < config.rotation_tolerance_rad
            )
            stalled = _stalled(errors, config)
            if within and (settled or stalled or steps >= config.max_motion_steps):
                break
            if self.ended:
                reason = self._ended_reason()
            elif steps >= config.max_motion_steps:
                reason = "max_steps"
            elif stalled:
                reason = "stalled"
            if reason is not None:
                break
            self._step(action)
        return ToolResult(
            data={
                "status": "reached" if reason is None else "partial",
                **({} if reason is None else {"reason": reason}),
                "env_steps": steps,
                "position_error_m": round(distance, 4),
                "rotation_error_rad": round(angle, 3),
            }
        )

    @tool
    def set_gripper(
        self, opening: Annotated[float, Field(ge=0.0, le=1.0)]
    ) -> ToolResult:
        """Drive the gripper to a normalized opening while holding the end-effector pose. Closing on an object stops early and reports the measured opening; contact does not imply a stable grasp. The target stays in effect during later moves. Returns reached/partial/error, measured pose/opening, residual opening error and environment steps. Example: {"opening":1}.

        Args:
            opening: Target opening from 0 (closed) to 1 (fully open).
        """
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        config = self._config
        hold = self._measure_native()
        hold_xyz = hold.xyz.copy()
        previous, still, reason = hold.opening, 0, None
        for steps in itertools.count():
            current = self._measure_native()
            error = opening - current.opening
            still = (
                still + 1
                if abs(current.opening - previous) < config.settle_opening
                else 0
            )
            previous = current.opening
            settled = still >= config.settle_steps
            within = abs(error) <= config.opening_tolerance
            if within and settled:
                break
            if self.ended:
                reason = self._ended_reason()
            elif steps >= config.max_gripper_steps:
                reason = "max_steps"
            elif settled and error < 0:
                reason = "blocked"
            if reason is not None:
                break
            action, _, _ = self._pose_action(current, hold_xyz, hold.rotation)
            self._command_gripper(action, opening, error, within)
            self._step(action)
        return ToolResult(
            data={
                "status": "reached" if reason is None else "partial",
                **({} if reason is None else {"reason": reason}),
                "env_steps": steps,
                "opening_error": round(error, 3),
            }
        )


class ManiSkillJointArm(ManiSkillArm):
    """Use selected channels of an absolute joint-position controller."""

    def __init__(
        self,
        episode,
        agent,
        tcp,
        mount,
        targets,
        channels,
        arm,
        fingers,
        fingertips,
        palm,
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
        self.fingertips = [self.robot.links_map[name] for name in fingertips]
        first, second = [transform(link.pose)[:3, 3] for link in self.fingertips]
        wrist = transform(self.robot.links_map[palm].pose)[:3, 3]
        tcp_rotation = self._tcp()[:3, :3].T
        jaw = first - second
        jaw /= np.linalg.norm(jaw)
        approach = (first + second) / 2 - wrist
        self.tool_rotation = grasp_axes(
            tcp_rotation @ jaw, tcp_rotation @ (approach - jaw * (jaw @ approach))
        )
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
            (f"{side}_two_link", f"{side}_four_link"),
            f"{side}_palm_link",
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
        ("thumb_tip", "index_tip"),
        "base",
    )


class ManiSkillRLPrimitives(RecordedPerception):
    """Drive the native robot through its Cartesian and mobile controllers."""

    use_requested_display_size = True
    BASE_LIMITS = np.array([0.35, 0.35, 0.5])

    @staticmethod
    def tool_profile(toolkit, perception):
        """Select the shared action batch and cameras supported by this robot."""
        if perception is None:
            cameras = CAMERAS["maniskill"]
            return ActionBatch(toolkit).execute_actions, (cameras,) * 3
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

    def __init__(self, *, env: LocalManiSkillEnv, max_episode_steps: int, **kwargs):
        self._init_perception(
            env=env,
            max_episode_steps=env.scene.cfg.init_params.max_episode_steps,
            **kwargs,
        )
        self._config = ControlConfig(position_step_m=0.07)
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
        self.surface_kind = "table" if mesh is not None else "floor"
        if mesh is None and "arena" in native.scene.actors:
            # Bridge builds its table and backdrop as one collision mesh.
            mount = transform(next(iter(self.arms.values())).mount.pose)
            for offset in ([0.15, 0.1, 0.3], [0.15, -0.1, 0.3], [0, 0.15, 0.3]):
                origin = mount[:3, 3] + mount[:3, :3] @ offset
                hit = native.scene.px.raycast(origin, [0, 0, -1], 2.0)
                if (
                    hit
                    and hit.component.entity == native.scene.actors["arena"]._objs[0]
                ):
                    self.surface_kind, self.surface = "table", hit.position
                    break
            else:
                raise ValueError(f"Could not measure the table in {self.env_id}")
        elif mesh is None:
            ground = getattr(native, "ground", None)
            if ground is None:
                ground = native.scene.actors.get("ground")
            if ground is None:
                raise ValueError(f"Missing support surface in {self.env_id}")
            # SAPIEN represents infinite floors with a plane collision.
            body = ground._objs[0].find_component_by_type(
                __import__("sapien").physx.PhysxRigidStaticComponent
            )
            plane = next(
                shape
                for shape in body.collision_shapes
                if type(shape).__name__ == "PhysxCollisionShapePlane"
            )
            self.surface = (ground._objs[0].pose * plane.local_pose).p
        else:
            self.surface = mesh.bounds[1]
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
            else:
                a, b = [transform(link.pose)[:3, 3] for link in arm.fingertips]
                measured["finger_gap_m"] = round(float(np.linalg.norm(a - b)), 5)
            states[name] = measured
        if self.mobile:
            arm = self.arms["arm"]
            states["arm"]["lift_m"] = float(
                np.asarray(arm.parts["body"].qpos)[0, self._lift_index]
            )
        budget = {
            "env_steps": self.env_steps,
            "env_steps_remaining": max(0, self._max_episode_steps - self.env_steps),
        }
        if self.dual_arm:
            return {**states, **budget}
        state = {**states["arm"], **budget}
        goal = self._obs.get("extra", {}).get("goal_pos")
        if goal is not None:
            state["goal"] = {
                "xyz": self.coordinates.points(np.asarray(goal)).round(4).tolist()
            }
        return state

    def _refresh_coordinates(self):
        mounts = [transform(arm.mount.pose) for arm in self.arms.values()]
        if self.dual_arm:
            calibration = mounting_calibration(
                mounts[0][:3, 3],
                mounts[1][:3, 3],
                self.surface,
                support_surface=self.surface_kind,
            )
        else:
            mount = mounts[0]
            calibration = {
                "base_xyz": mount[:3, 3],
                "base_rotation": np.roll(
                    mount[:3, :3], -self._mount_forward_axis, axis=1
                ),
                f"{self.surface_kind}_top_xyz": self.surface,
            }
        calibration["tcp"] = (
            "native_grasp_tcp" if self.has_gripper else "native_tool_tip"
        )
        coords = Coordinates.from_control({"coordinate_calibration": calibration})
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

    def _mobility_delta(self, before):
        after = self._mobility_state()
        initial_frame = before[1].inv()
        displacement = initial_frame.apply(after[0] - before[0])
        forward = (initial_frame * after[1]).as_matrix()[:, 0]
        yaw = float(np.arctan2(forward[1], forward[0]))
        return displacement, yaw, float(after[2][2] - before[2][2])

    def _mobility_feedback(self, motion, start_steps, request, reason):
        displacement, yaw, height_change = motion
        return {
            "status": "reached" if reason == "completed" else "partial",
            "reason": reason,
            "requested": request,
            "displacement_initial_base_m": np.round(displacement, 4).tolist(),
            "yaw_change_rad": round(yaw, 4),
            "mount_height_change_m": round(height_change, 4),
            "executed_steps": self.env_steps - start_steps,
            "executed_duration_s": round(
                (self.env_steps - start_steps) / self._meta["control_freq"], 4
            ),
            "env_steps": self.env_steps,
        }

    @tool
    def move_base(
        self,
        vx: Annotated[float, Field(ge=-0.35, le=0.35)] = 0,
        vy: Annotated[float, Field(ge=-0.35, le=0.35)] = 0,
        yaw_rate: Annotated[float, Field(ge=-0.5, le=0.5)] = 0,
        duration_s: Annotated[float, Field(ge=0.1, le=2)] = 1,
    ) -> ToolResult:
        """Move the mobile base at bounded velocity in its current local frame, then stop."""
        physical = np.asarray([vx, vy, yaw_rate], dtype=float)
        cfg = self._meta["parts"]["base"]
        if (
            not np.isfinite(physical).all()
            or not np.isfinite(duration_s)
            or not 0.1 <= duration_s <= 2
            or np.any(np.abs(physical) > self.BASE_LIMITS)
            or np.any(physical < cfg["actuator_min"])
            or np.any(physical > cfg["actuator_max"])
        ):
            return ToolResult(
                data={"error": "Invalid mobile-base velocity or duration"}
            )
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        before, start_steps = self._mobility_state(), self.env_steps
        requested_steps = max(2, round(duration_s * self._meta["control_freq"]))
        count = min(requested_steps, self._max_episode_steps - self.env_steps)
        reason, drive_steps = "completed", 0
        try:
            for _ in range(max(0, count - 1)):
                if self.ended:
                    reason = "episode_ended"
                    break
                self._mobile_step(physical)
                drive_steps += 1
        finally:
            if not self.ended and self.env_steps < self._max_episode_steps:
                self._mobile_step(check_cancelled=False)
        if self.ended:
            reason = "episode_ended" if self.native_success() else "budget_exhausted"
        elif count < requested_steps:
            reason = "budget_exhausted"
        motion = self._mobility_delta(before)
        displacement, yaw, _ = motion
        result = self._mobility_feedback(
            motion,
            start_steps,
            {"vx": vx, "vy": vy, "yaw_rate": yaw_rate, "duration_s": duration_s},
            reason,
        )
        duration = drive_steps / self._meta["control_freq"]
        expected_yaw = yaw_rate * duration
        half_yaw = expected_yaw / 2
        velocity = Rotation.from_euler("z", half_yaw).apply([vx, vy, 0])[:2]
        expected_xy = duration * np.sinc(half_yaw / np.pi) * velocity
        residual_xy = expected_xy - displacement[:2]
        expected_distance = float(np.linalg.norm(expected_xy))
        translation_error = float(np.linalg.norm(residual_xy))
        yaw_error = expected_yaw - yaw
        result["translation_error_m"] = round(translation_error, 4)
        result["translation_shortfall_m"] = round(
            float(residual_xy @ expected_xy) / expected_distance
            if expected_distance
            else 0.0,
            4,
        )
        result["yaw_shortfall_rad"] = round(float(np.sign(expected_yaw) * yaw_error), 4)
        if reason == "completed":
            result["status"] = "commanded"
            if translation_error > max(0.05, expected_distance * 0.25) or abs(
                yaw_error
            ) > max(0.0524, abs(expected_yaw) * 0.25):
                result.update(status="partial", reason="limited_motion")
        return ToolResult(data=result)

    @tool
    def move_lift(
        self,
        delta_m: Annotated[float, Field(ge=-0.25, le=0.25)],
        duration_s: Annotated[float, Field(ge=0.1, le=5)] = 3,
    ) -> ToolResult:
        """Raise or lower the arm's mounting support relative to its current measured height."""
        if (
            not np.isfinite([delta_m, duration_s]).all()
            or not -0.25 <= delta_m <= 0.25
            or not 0.1 <= duration_s <= 5
        ):
            return ToolResult(data={"error": "Invalid lift displacement or duration"})
        if self.ended:
            return ToolResult(data={"error": "episode_ended"})
        before, start_steps = self._mobility_state(), self.env_steps
        target = before[2][2] + delta_m
        count = min(
            max(2, round(duration_s * self._meta["control_freq"])),
            self._max_episode_steps - self.env_steps,
        )
        reason, previous, still = "motion_budget", before[2][2], 0
        cfg = self._meta["parts"]["torso"]
        try:
            for _ in range(max(0, count - 1)):
                if self.ended:
                    reason = "episode_ended"
                    break
                height = self._mobility_state()[2][2]
                error = target - height
                if abs(error) <= 0.004:
                    reason = "completed"
                    break
                still = still + 1 if abs(height - previous) < 0.0002 else 0
                previous = height
                if still >= 12:
                    reason = "blocked_or_joint_limit"
                    break
                low, high = float(cfg["output_min"][0]), float(cfg["output_max"][0])
                self._mobile_step(
                    lift=float(np.clip(error, max(low, -0.025), min(high, 0.025)))
                )
        finally:
            if not self.ended and self.env_steps < self._max_episode_steps:
                self._mobile_step(check_cancelled=False)
        motion = self._mobility_delta(before)
        height_error = delta_m - motion[2]
        if self.ended:
            reason = "episode_ended" if self.native_success() else "budget_exhausted"
        elif abs(height_error) <= 0.004:
            reason = "completed"
        elif reason == "completed":
            reason = "outside_tolerance_after_stop"
        result = self._mobility_feedback(
            motion, start_steps, {"delta_m": delta_m, "duration_s": duration_s}, reason
        )
        result["height_error_m"] = round(height_error, 4)
        return ToolResult(data=result)

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
        native = self.env.scene.unwrapped
        mode = native.obs_mode
        try:
            if not self.record_artifacts:
                native._obs_mode = "state_dict"
            self._obs, reward, terminated, truncated, self._info = self.env.scene.step(
                action
            )
        finally:
            native._obs_mode = mode
        self.terminated, self.truncated = (
            bool(terminated),
            bool(truncated) and not bool(terminated),
        )
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
        if "sensor_data" not in self._obs:
            scene = self.env.scene
            self._obs = scene._wrap_obs(scene.unwrapped.get_obs())
        data = self._obs["sensor_data"][camera]
        rgb = data["rgb"]
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
        if camera not in self.depth_cameras:
            return (
                rgb,
                np.full((*rgb.shape[:2], 3), np.nan, dtype=np.float32),
                calibration,
            )
        # ManiSkill reports axial depth in millimetres, zero marks missing pixels.
        depth = np.asarray(data["depth"], dtype=np.float32) / 1000
        if depth.ndim == 3:
            depth = depth.squeeze(axis=-1)
        depth[depth <= 0] = np.nan
        points = world_from_depth(depth, calibration).astype(np.float32)
        return rgb, points, calibration

    @property
    def segment(self):
        return super().segment.with_handler(self._segment_with_table_reference)

    def _segment_with_table_reference(self, **kwargs):
        result = super().segment(**kwargs)
        mask_id = result.data.get("mask_id")
        if mask_id is None:
            return result
        step, camera, mask = self._masks[mask_id]
        if camera in self.depth_cameras or self.coordinates.table_height_m is None:
            return result
        capture = self._steps[step].cameras[camera]
        rows, cols = np.nonzero(mask)
        # Select a real mask pixel nearest its median, including for concave masks.
        i = np.argmin((rows - np.median(rows)) ** 2 + (cols - np.median(cols)) ** 2)
        plane = self._table_reference(capture.calibration, rows[i], cols[i])
        if plane is not None:
            result.data["table_reference"] = plane
        return result

    def _table_reference(self, calibration, row, col):
        """Intersect a calibrated camera ray with the measured support plane."""
        pose = np.asarray(calibration["extrinsic_cam2world"])
        ray = pose[:3, :3] @ np.linalg.solve(calibration["intrinsic_K"], [col, row, 1])
        direction = self.coordinates.rotation.T @ ray
        origin = self.coordinates.points(pose[:3, 3])
        if abs(direction[2]) < 1e-6:
            return None
        distance = (self.coordinates.table_height_m - origin[2]) / direction[2]
        if distance <= 0:
            return None
        return {
            "xyz": (origin + distance * direction).round(5).tolist(),
            "xy_per_height": (direction[:2] / direction[2]).round(4).tolist(),
        }

# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""One native ManiSkill Gymnasium episode exposed through the RPent toolkit."""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import Field

from robots.rl.actions import ActionBatch, DualArmBatch, MobileBatch, MoveAction
from robots.rl.backends import RoboCasaGeneralPrimitives
from robots.rl.coordinates import Coordinates, dual_arm_mount, horizontal_rotation
from robots.rl.geometry import world_from_depth
from robots.rl.maniskill_arm import (
    ManiSkillArm,
    UnsupportedRobotPolicy,
    _normalize_target,
    floating_hand,
    g1_arms,
    transform,
)
from robots.rl.perception import RecordedPerception
from robots.rl.tools import ControlConfig, LiberoRLPrimitives
from rpent.session import EnvState
from rpent.tools import ToolResult, tool


class ToolTipBatch(ActionBatch):
    @tool
    def execute_actions(
        self, actions: Annotated[list[MoveAction], Field(min_length=1)]
    ) -> ToolResult:
        """Execute 1..K move_to items for the tool tip. Each item is measured before continuing."""
        return self._execute_actions(actions)


def tool_profile(toolkit, capabilities):
    """Select the shared action batch and cameras supported by this robot."""
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


def remote_capabilities(metadata):
    """Check the server's measured TCP before binding tools to the RPC client."""
    capabilities = metadata["tool_capabilities"]
    coords = metadata.get("coordinates") or {}
    tcp = "native_grasp_tcp" if capabilities["gripper"] else "native_tool_tip"
    if (
        coords.get("contract_version") != "episode_coordinates_v2"
        or coords.get("tcp") != tcp
    ):
        raise ValueError(
            "remote ManiSkill TCP or mounting frame differs from the planner"
        )
    return capabilities


def recorded_outcome(state):
    """Read native verdicts and return from the last transferred observation."""
    extras = state.latest_record().extras
    return {
        key: extras[key]
        for key in ("success", "fail", "episode_return", "terminated", "truncated")
    }


def scored_outcome(outcome: dict, native: dict) -> dict:
    """Keep live simulator metrics, including steps after the last observation."""
    outcome = {**outcome, **native}
    if (
        outcome["termination_reason"] != "infrastructure_error"
        and not native["success"]
    ):
        if native["terminated"]:
            outcome["termination_reason"] = (
                "environment_failure" if native["fail"] else "environment_terminated"
            )
            outcome["terminal"] = True
        elif native["truncated"]:
            outcome.update(termination_reason="environment_step_limit", terminal=True)
    return outcome


def _task_hint(env_id):
    instructions = (
        Path(__file__).with_name("prompts") / "maniskill_tasks.md"
    ).read_text(encoding="utf-8")
    section = instructions.partition(f"\n## {env_id}\n")[2].partition("\n## ")[0]
    return " ".join(section.split())


def _json(value):
    if isinstance(value, dict):
        return {name: _json(item) for name, item in value.items()}
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.item() if value.numel() == 1 else value.tolist()
    return value


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
    env.action_space = env.env.action_space
    env.spec = env.env.spec
    return env, env_id


class ManiSkillPrimitives(RecordedPerception):
    """Expose native policy observations, controller actions and episode metrics."""

    # Fetch uses the existing base/lift loops with its own measured state and native step.
    BASE_LIMITS = RoboCasaGeneralPrimitives.BASE_LIMITS
    move_base = RoboCasaGeneralPrimitives.move_base
    move_lift = RoboCasaGeneralPrimitives.move_lift
    _mobility_delta = RoboCasaGeneralPrimitives._mobility_delta
    _mobility_feedback = RoboCasaGeneralPrimitives._mobility_feedback
    _save = LiberoRLPrimitives._save

    def __init__(
        self,
        *,
        task_name: str,
        seed: int,
        state: EnvState,
        check_cancelled: Callable[[], None],
        sam3_client=None,
        render_device: int | None = None,
        max_episode_steps: int | None = None,
    ) -> None:
        self._state = state
        self._sam3 = sam3_client
        self._masks = {}
        self._mask_ids = itertools.count(1)
        self._sam_calls = itertools.count(1)
        self._check_cancelled = check_cancelled
        self._config = ControlConfig()
        self.frames = []
        self.env, self.env_id = _make_env(
            task_name, render_device, max_episode_steps, seed
        )
        try:
            self._obs, self._info = self.env.reset(seed=seed)
            sensors = self._obs.get("sensor_data", {})
            params = self._obs.get("sensor_param", {})
            self.cameras = tuple(
                name for name, data in sensors.items() if "rgb" in data
            )
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
            native = self.env.unwrapped
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
                self._lift_index = [
                    joint.name for joint in arm.parts["body"].joints
                ].index("torso_lift_joint")
                self._max_episode_steps = (
                    self.env.cfg.init_params.max_episode_steps or float("inf")
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
            state.save("coordinates.json", self.coordinates.metadata(), step=None)
            self.task = self.env.instruction
            if not self.task:
                doc = inspect.cleandoc(type(native).__doc__ or "")
                description = (
                    doc.removeprefix("**Task Description:**").strip().split("\n\n")[0]
                )
                self.task = (
                    f"{self.env_id}: {description}" if description else self.env_id
                )
            if hint := _task_hint(self.env_id):
                if "{max_episode_steps}" in hint:
                    hint = hint.format(
                        max_episode_steps=self.env.cfg.init_params.max_episode_steps
                    )
                self.task += " " + hint
        except BaseException:
            self.env.close()
            raise

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
            state["cube_size_m"] = 2 * self.env.unwrapped.cube_half_size
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
        self._obs, reward, terminated, truncated, self._info = self.env.step(action)
        self.terminated, self.truncated = bool(terminated), bool(truncated)
        self.reward = float(reward)
        episode = self._info["episode"]
        self.episode_return = float(episode["return"])
        self.env_steps = int(episode["episode_len"])

        self._record_video_frame()
        self._refresh_coordinates()

    def _record_video_frame(self):
        if self.cameras:
            self.frames.append(self._obs["sensor_data"][self.cameras[0]]["rgb"].copy())

    def record_observation(self, *, command=None, result=None, elapsed_s=None):
        if self.env_steps == 0 and not self.frames:
            self._record_video_frame()
        with self._state.record_step(
            state=self.robot_state(),
            terminated=self.terminated,
            truncated=self.truncated,
            command=command,
            result=result,
            elapsed_s=elapsed_s,
            extras={
                **self.native_outcome(),
                "env_steps": self.env_steps,
                "reward": self.reward,
                "episode": _json(self._info.get("episode", {})),
            },
        ) as step:
            self._save("step_coordinates.json", self.coordinates.metadata(), step)
            for camera in self.cameras:
                self._record_camera(camera, step)
        return self._state.get(step)

    def _record_camera(self, camera, step):
        data = self._obs["sensor_data"][camera]
        self._save(f"{camera}.png", data["rgb"], step)
        if camera in self.depth_cameras:
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
            self._save(f"{camera}_world.npz", points, step)
            self._save(f"{camera}_calibration.json", calibration, step)

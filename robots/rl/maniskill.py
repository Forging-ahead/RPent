# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""One native ManiSkill Gymnasium episode exposed through the RPent toolkit."""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable
from typing import Any

import numpy as np

from robots.rl.backends import MeasuredMobileMotion
from robots.rl.coordinates import Coordinates
from robots.rl.geometry import world_from_depth
from robots.rl.maniskill_arm import ManiSkillArm, UnsupportedRobotPolicy, transform
from robots.rl.perception import RecordedPerception
from robots.rl.tools import ControlConfig
from rpent.session import EnvState, StepRecord


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
    from mani_skill.utils import common
    from mani_skill.utils.registration import REGISTERED_ENVS
    from omegaconf import OmegaConf
    from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv

    env_id = "PickCube-v1" if task_name == "pick_cube" else task_name
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
    env.action_space = env.env.action_space
    env.spec = env.env.spec
    return env, env_id


class ManiSkillPrimitives(MeasuredMobileMotion, RecordedPerception):
    """Expose native policy observations, controller actions and episode metrics."""

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
        self._check_cancelled = check_cancelled
        self._sam3 = sam3_client
        self._config = ControlConfig()
        self._sam_calls = itertools.count(1)
        self._mask_ids = itertools.count(1)
        self._masks = {}
        self.frames = []
        self.env, self.env_id = _make_env(
            task_name, render_device, max_episode_steps, seed
        )
        try:
            self._obs, self._info = self.env.reset(seed=seed)
            self.env_steps = 0
            self.episode_return = 0.0
            self.reward = None
            self.terminated = self.truncated = False
            native = self.env.unwrapped
            group = getattr(native.agent, "agents_dict", None)
            self.dual_arm = group is not None
            if self.dual_arm:
                if (
                    len(group) != 2
                    or not hasattr(native, "left_agent")
                    or not hasattr(native, "right_agent")
                ):
                    raise UnsupportedRobotPolicy(
                        f"{self.env_id} needs a physical arm mapping"
                    )
                self.arms = {
                    side: ManiSkillArm(self, getattr(native, f"{side}_agent"))
                    for side in ("left", "right")
                }
                self.agent_keys = {
                    side: next(k for k, v in group.items() if v is arm.agent)
                    for side, arm in self.arms.items()
                }
            else:
                self.arms = {"arm": ManiSkillArm(self, native.agent)}
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
            self.has_gripper = all(arm.has_gripper for arm in self.arms.values())
            table = getattr(getattr(native, "table_scene", None), "table", None)
            mesh = table.get_first_collision_mesh() if table is not None else None
            self.table_z = None if mesh is None else float(mesh.bounds[1, 2])
            self._refresh_coordinates()
            state.save("coordinates.json", self.coordinates.metadata(), step=None)
            instruction = getattr(native, "get_language_instruction", None)
            if instruction is not None:
                value = instruction()
                self.task = value if isinstance(value, str) else value[0]
            else:
                doc = inspect.cleandoc(type(native).__doc__ or "")
                description = (
                    doc.removeprefix("**Task Description:**").strip().split("\n\n")[0]
                )
                self.task = (
                    f"{self.env_id}: {description}" if description else self.env_id
                )
        except BaseException:
            self.env.close()
            raise

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, data in self._obs.get("sensor_data", {}).items()
            if "rgb" in data
        )

    @property
    def depth_cameras(self) -> tuple[str, ...]:
        params = self._obs.get("sensor_param", {})
        return tuple(
            name
            for name in self.cameras
            if "depth" in self._obs["sensor_data"][name]
            and {"intrinsic_cv", "extrinsic_cv"} <= params.get(name, {}).keys()
        )

    @property
    def ended(self) -> bool:
        return self.terminated or self.truncated

    def _verdict(self, name: str) -> bool | None:
        metrics = self._info.get("episode", {})
        value = metrics.get(f"{name}_once", self._info.get(name))
        return None if value is None else bool(value)

    def native_success(self) -> bool | None:
        return self._verdict("success")

    def native_fail(self) -> bool | None:
        return self._verdict("fail")

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
        return states if self.dual_arm else states["arm"]

    def _refresh_coordinates(self):
        mounts = [transform(arm.mount.pose) for arm in self.arms.values()]
        origin = np.mean([m[:3, 3] for m in mounts], axis=0)
        up = np.array([0.0, 0.0, 1.0])
        if self.dual_arm:
            forward = np.cross(mounts[0][:3, 3] - mounts[1][:3, 3], up)
        else:
            forward = mounts[0][:3, 0].copy()
        forward[2] = 0
        if np.linalg.norm(forward) < 1e-6:
            raise ValueError("Cannot determine horizontal arm mounting frame")
        forward /= np.linalg.norm(forward)
        coords = Coordinates(
            origin,
            np.column_stack((forward, np.cross(up, forward), up)),
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
        base = arm.parts["base"]._original_single_action_space
        body = arm.parts["body"]._original_single_action_space
        twist = np.zeros(2) if velocity is None else np.asarray(velocity)[[0, 2]]
        command[arm.slices["base"]] = (
            2 * (twist - base.low) / (base.high - base.low) - 1
        )
        delta = np.zeros(body.shape)
        delta[self._lift_index] = lift
        command[arm.slices["body"]] = (
            2 * (delta - body.low) / (body.high - body.low) - 1
        )
        arm.last = command
        self.step_controllers(check_cancelled=check_cancelled)

    def step_controllers(self, *, check_cancelled=True) -> None:
        if check_cancelled:
            self._check_cancelled()
        if self.ended:
            raise RuntimeError("ManiSkill episode has ended")
        # Explicit batch dimensions remain valid after an official control-mode switch.
        if self.dual_arm:
            action = {
                self.agent_keys[k]: arm.last[None, :] for k, arm in self.arms.items()
            }
        else:
            action = self.arms["arm"].last[None, :]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs, self._info = obs, info
        self.terminated, self.truncated = bool(terminated), bool(truncated)
        self.reward = float(reward)
        self.episode_return = float(info["episode"]["return"])
        self.env_steps = int(info["episode"]["episode_len"])

        self._refresh_coordinates()

    def _save(self, name: str, value: Any, step: int) -> None:
        if self._state.save(name, value, step=step) is None:
            raise RuntimeError(f"failed to save observation artifact {name}")

    def record_observation(
        self, *, command=None, result=None, elapsed_s=None
    ) -> StepRecord:
        extras = {
            "env_steps": self.env_steps,
            "reward": self.reward,
            "episode_return": self.episode_return,
            "episode": _json(self._info.get("episode", {})),
            "success": self.native_success(),
            "fail": self.native_fail(),
            "terminated": self.terminated,
            "truncated": self.truncated,
        }
        with self._state.record_step(
            state=self.robot_state(),
            command=command,
            result=result,
            elapsed_s=elapsed_s,
            terminated=self.terminated,
            truncated=self.truncated,
            extras=extras,
        ) as step:
            self._save("step_coordinates.json", self.coordinates.metadata(), step)
            for i, camera in enumerate(self.cameras):
                data = self._obs["sensor_data"][camera]
                rgb = data["rgb"]
                self._save(f"{camera}.png", rgb, step)
                if i == 0:
                    self.frames.append(rgb)
                if camera not in self.depth_cameras:
                    continue
                params = self._obs["sensor_param"][camera]
                # ManiSkill emits OpenCV world-to-camera extrinsics and axial depth in millimetres.
                extrinsic = np.eye(4)
                extrinsic[:3] = np.asarray(params["extrinsic_cv"])[:3, :4]
                calibration = {
                    "intrinsic_K": params["intrinsic_cv"].tolist(),
                    "extrinsic_cam2world": np.linalg.inv(extrinsic).tolist(),
                }
                depth = np.asarray(data["depth"])
                if depth.ndim == 3:
                    depth = depth.squeeze(axis=-1)
                depth = depth.astype(np.float32) / 1000.0
                depth[depth <= 0] = np.nan
                points = world_from_depth(depth, calibration).astype(np.float32)
                self._save(f"{camera}_world.npz", points, step)
                self._save(f"{camera}_calibration.json", calibration, step)
        return self._state.get(step)

# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""One native ManiSkill Gymnasium episode exposed through the RPent toolkit."""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable
from typing import Any

import numpy as np

from robots.rl.geometry import world_from_depth
from robots.rl.perception import RecordedPerception
from robots.rl.tools import ControlConfig
from rpent.session import EnvState, StepRecord


def _json(value):
    if isinstance(value, dict):
        return {name: _json(item) for name, item in value.items()}
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def _make_env(task_name: str, render_device: int | None, max_episode_steps: int | None):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - imports native task registrations
    from mani_skill.utils.registration import REGISTERED_ENVS
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

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
    try:
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode=mode,
            sim_backend="physx_cpu",
            render_backend="gpu" if render_device is None else f"cuda:{render_device}",
            max_episode_steps=horizon,
        )
    except (EOFError, SystemExit) as e:
        raise RuntimeError(
            f"ManiSkill could not initialize {env_id}; check its required assets"
        ) from e
    try:
        env = CPUGymWrapper(env, record_metrics=True)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        return env, env_id
    except BaseException:
        env.close()
        raise


def _bounds(values: np.ndarray) -> list[float | None]:
    """Keep unbounded action dimensions representable in JSON."""
    return [float(v) if np.isfinite(v) else None for v in values]


def _action_description(env) -> dict:
    """Describe the actual Gym space and the ordering used by the native controllers."""
    agent = env.unwrapped.agent
    controllers = getattr(agent, "agents_dict", None)
    if controllers is None:
        ordered = [(agent.uid, agent)]
    else:
        ordered = [
            (name, controllers[name])
            for name in env.unwrapped.single_action_space.spaces
        ]
    parts = []
    offset = 0
    for robot, owner in ordered:
        controller = owner.controller
        width = owner.single_action_space.shape[0]
        groups = getattr(controller, "action_mapping", {"action": (0, width)})
        for group, (begin, end) in groups.items():
            if begin == end:
                continue
            component = getattr(controller, "controllers", {}).get(group, controller)
            part = {
                "robot": robot,
                "group": group,
                "slice": [offset + begin, offset + end],
                "controller": type(component).__name__,
                "joints": [joint.name for joint in component.joints],
            }
            for key in ("normalize_action", "use_delta", "use_target", "frame"):
                if hasattr(component.config, key):
                    part[key] = getattr(component.config, key)
            original_space = getattr(component, "_original_single_action_space", None)
            if original_space is not None:
                part["controller_low"] = _bounds(original_space.low)
                part["controller_high"] = _bounds(original_space.high)
            parts.append(part)
        offset += width
    space = env.action_space

    return {
        "shape": list(space.shape),
        "low": _bounds(space.low),
        "high": _bounds(space.high),
        "control_mode": agent.control_mode,
        "components": parts,
    }


class ManiSkillPrimitives(RecordedPerception):
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
        self.env, self.env_id = _make_env(task_name, render_device, max_episode_steps)
        try:
            self._obs, self._info = self.env.reset(seed=seed)
            self.env_steps = 0
            self.episode_return = 0.0
            self.reward = None
            self.terminated = self.truncated = False
            self.action_spec = _action_description(self.env)
            native = self.env.unwrapped
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
        """Only expose policy observation fields, never simulator or evaluation ground truth."""
        return _json(
            {
                name: value
                for name, value in self._obs.items()
                if name not in {"sensor_data", "sensor_param"}
            }
        )

    def validate_action(self, action) -> np.ndarray:
        values = np.asarray(action)
        space = self.env.action_space
        if values.shape != space.shape or values.dtype.kind not in "fiu":
            raise ValueError(f"Expected {space.shape[0]} real controller inputs")
        if (
            not np.isfinite(values).all()
            or np.any(values < space.low)
            or np.any(values > space.high)
        ):
            raise ValueError(
                "Controller inputs must be finite and within the native action bounds"
            )
        return values.astype(space.dtype)

    def step_action(self, action) -> None:
        self._check_cancelled()
        if self.ended:
            raise RuntimeError("ManiSkill episode has ended")
        obs, reward, terminated, truncated, info = self.env.step(
            self.validate_action(action)
        )
        self._obs, self._info = obs, info
        self.terminated, self.truncated = bool(terminated), bool(truncated)
        self.reward = float(reward)
        self.episode_return = float(info["episode"]["return"])
        self.env_steps = int(info["episode"]["episode_len"])

    @staticmethod
    def _model_points(world: np.ndarray) -> np.ndarray:
        return world

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

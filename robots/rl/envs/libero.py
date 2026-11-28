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

"""LIBERO single-arm Panda general-frame adapter and in-process warm host."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from robots.rl import geometry
from robots.rl.control import CartesianServo, ControlConfig, Measurement, _clip_norm
from robots.rl.geometry import Coordinates, grasp_axes
from robots.rl.perception import RecordedPerception

SIM_CAMERAS: dict[str, str] = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
IMAGE_SIZE = 1024


class LocalLiberoEnv:
    """In-process LIBERO host exposing the same surface as LiberoEnvClient.

    Eliminates the extra env_server HTTP process and loopback RGB-D serialization.
    """

    def __init__(
        self,
        *,
        suite: str,
        task: int,
        seed: int,
        libero_type: str | None = None,
        max_episode_steps: int = 10000,
        record_artifacts: bool = False,
    ) -> None:
        from robots.libero.env_server import LiberoEnvFacade, make_env
        from robots.libero.suites import resolve_libero_type

        os.environ["LIBERO_TYPE"] = resolve_libero_type(suite, libero_type)
        # Skip per-step offscreen camera rendering during training physics loops;
        # OffScreenRenderEnv keeps has_offscreen_renderer=True so capture() still
        # renders RGB-D on demand via render_camera().
        raw_env = make_env(
            task_id=int(task),
            seed=int(seed),
            suite_name=suite,
            max_episode_steps=int(max_episode_steps),
            rl_control_state=True,
            use_camera_obs=bool(record_artifacts),
            camera_depths=False,
        )
        self._facade = LiberoEnvFacade(
            raw_env,
            meta={
                "suite": suite,
                "task": int(task),
                "seed": int(seed),
                "max_episode_steps": int(max_episode_steps),
            },
        )
        obs, info = self._facade.reset()
        self.last_obs: dict[str, Any] = obs
        self.last_info: dict[str, Any] = info
        self.terminated = False
        self.truncated = False

    def step(self, action: np.ndarray) -> None:
        obs, _rew, term, trunc, info = self._facade.step(action)
        self.last_obs = obs
        self.last_info = info
        self.terminated = bool(term)
        self.truncated = bool(trunc)

    def render_camera(
        self,
        camera_name: str,
        height: int = IMAGE_SIZE,
        width: int = IMAGE_SIZE,
        depth: bool = False,
    ) -> Any:
        return self._facade.render_camera(
            camera_name=camera_name, height=height, width=width, depth=depth
        )

    def get_camera_meta(
        self, camera_name: str, height: int = IMAGE_SIZE, width: int = IMAGE_SIZE
    ) -> dict[str, Any] | None:
        return self._facade.get_camera_meta(
            camera_name=camera_name, height=height, width=width
        )

    def get_task_language(self) -> str:
        return str(self._facade.get_task_language() or "")

    def close(self) -> None:
        self._facade.close()


class LiberoRLPrimitives(CartesianServo, RecordedPerception):
    """Servo one LIBERO Franka Panda arm and query recorded RGB-D views."""

    cameras = tuple(SIM_CAMERAS)

    def __init__(
        self,
        *,
        env: Any,
        sam3_client: Any = None,
        state: Any = None,
        check_cancelled,
        config: ControlConfig = ControlConfig(),
        max_episode_steps: int = 10000,
        record_artifacts: bool = False,
    ) -> None:
        self._init_perception(
            env=env,
            sam3_client=sam3_client,
            state=state,
            check_cancelled=check_cancelled,
            max_episode_steps=max_episode_steps,
            record_artifacts=record_artifacts,
            segment_min_score=config.segment_min_score,
        )
        self._config = config
        control = env.last_obs["control_state"]
        self.coordinates = Coordinates.from_control(control)
        self.tool_rotation = grasp_axes(
            control["jaw_slide_axes_site"][0], control["wrist_to_site_in_site"]
        )
        self._save_artifact("coordinates.json", self.coordinates.metadata(), None)
        self._open_gap = self._finger_gap()
        if self.record_artifacts and "main_images" in env.last_obs:
            self.frames = [np.asarray(env.last_obs["main_images"])]

    @property
    def ended(self) -> bool:
        return self.env.terminated or self.env.truncated

    def native_success(self) -> bool:
        return bool(self.env.terminated)

    def _ended_reason(self) -> str:
        return "terminated" if self.env.terminated else "truncated"

    def _finger_gap(self) -> float:
        states = np.asarray(self.env.last_obs["states"], dtype=np.float64)
        return float(abs(states[6]) + abs(states[7]))

    def _measure_native(self) -> Measurement:
        control = self.env.last_obs["control_state"]
        return Measurement(
            xyz=np.asarray(control["xyz"], dtype=np.float64),
            rotation=np.asarray(control["rotation"], dtype=np.float64),
            opening=min(1.0, self._finger_gap() / self._open_gap),
        )

    def robot_state(self) -> dict[str, Any]:
        return {
            **self.measure().state(),
            "finger_gap_m": round(self._finger_gap(), 5),
        }

    def capture(self, camera: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        sim_camera = SIM_CAMERAS[camera]
        rgb, depth = self.env.render_camera(
            sim_camera, IMAGE_SIZE, IMAGE_SIZE, depth=True
        )
        meta = self.env.get_camera_meta(sim_camera, IMAGE_SIZE, IMAGE_SIZE)
        if meta is None:
            raise RuntimeError(f"{sim_camera} camera metadata is unavailable")
        metric = geometry.metric_depth(depth, meta)[::-1]
        world = geometry.world_from_depth(metric, meta)
        valid = (
            np.isfinite(metric)
            & (metric > meta["depth_near"] * (1 + 1e-4))
            & (metric < meta["depth_far"] * (1 - 1e-5))
        )
        world[~valid] = np.nan
        calibration = {
            **meta,
            "coordinates": self.coordinates.metadata(),
            "tool_rotation_to_native": self.tool_rotation.tolist(),
        }
        return np.asarray(rgb)[::-1], world, calibration

    def _step(self, action: np.ndarray) -> None:
        self._check_cancelled()
        self.env.step(action)
        self.env_steps += 1
        if self.record_artifacts and "main_images" in self.env.last_obs:
            self.frames.append(np.asarray(self.env.last_obs["main_images"]))

    def _pose_action(
        self, current: Measurement, xyz: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        position_error = xyz - current.xyz
        rotation_error = Rotation.from_matrix(rotation @ current.rotation.T).as_rotvec()
        action = np.zeros(7)
        control = self.env.last_obs["control_state"]
        frame = np.asarray(control["control_to_world_rotation"], dtype=np.float64)
        delta = np.concatenate(
            [
                frame.T @ _clip_norm(position_error, self._config.position_step_m),
                frame.T @ _clip_norm(rotation_error, self._config.rotation_step_rad),
            ]
        )
        low, high = np.asarray(control["output_min"]), np.asarray(control["output_max"])
        input_low, input_high = (
            np.asarray(control["input_min"]),
            np.asarray(control["input_max"]),
        )
        if np.any(high <= low):
            raise ValueError("native OSC controller has invalid output bounds")
        action[:6] = (delta - (high + low) / 2) * (input_high - input_low) / (
            high - low
        ) + (input_high + input_low) / 2
        if np.any(action[:6] < input_low - 1e-8) or np.any(
            action[:6] > input_high + 1e-8
        ):
            raise ValueError("servo step exceeds the native controller bounds")
        return (
            action,
            float(np.linalg.norm(position_error)),
            float(np.linalg.norm(rotation_error)),
        )

    def _command_gripper(self, action, opening, error, within):
        action[6] = 0.0 if within else float(-np.sign(error))

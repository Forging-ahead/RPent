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

"""Shared measured Cartesian servo and mobile base commands."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from robots.rl import geometry
from rpent.tools import ToolResult, tool


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
    segment_min_score: float = 0.2


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


class CartesianServo:
    """Pose and gripper control using native measurements and commands."""

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


class MobileControl:
    """Bounded chassis and lift commands using native odometry."""

    BASE_LIMITS = np.array([0.35, 0.35, 0.5])

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

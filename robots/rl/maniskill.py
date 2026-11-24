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

"""Native ManiSkill PickCube backend for the shared episode tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from robots.rl import geometry
from robots.rl.backends import ScenePrimitives
from robots.rl.coordinates import Coordinates, grasp_axes, pose_matrix
from robots.rl.tools import IMAGE_SIZE, Measurement
from rpent.session import EnvState

if TYPE_CHECKING:
    from rpent.robots.components.sam3_client import Sam3Client


class ManiSkillPrimitives(ScenePrimitives):
    """Adapt a single Panda scene to RPent's measured pose and RGB-D contract."""

    cameras = ("agentview",)

    def __init__(
        self,
        *,
        task_name: str,
        seed: int,
        render_device: int | None,
        sam3_client: Sam3Client,
        state: EnvState,
        check_cancelled: Callable[[], None],
        max_episode_steps: int,
    ) -> None:
        if task_name != "pick_cube":
            raise ValueError("ManiSkill currently supports task_name=pick_cube")
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 - registers native Gymnasium tasks

        env = gym.make(
            "PickCube-v1",
            num_envs=1,
            obs_mode="rgbd",
            control_mode="pd_ee_pose",
            sim_backend="physx_cpu",
            render_backend="gpu" if render_device is None else f"cuda:{render_device}",
            sensor_configs={"base_camera": {"width": IMAGE_SIZE, "height": IMAGE_SIZE}},
            max_episode_steps=max_episode_steps,
        )
        try:
            super().__init__(
                env=env,
                sam3_client=sam3_client,
                state=state,
                check_cancelled=check_cancelled,
                max_episode_steps=max_episode_steps,
            )
            self._obs, _ = env.reset(seed=seed)
            self._agent = env.unwrapped.agent
            self._base = pose_matrix(self._agent.robot.pose.raw_pose[0].cpu().numpy())
            gripper = self._agent.controller.controllers["gripper"].config
            self._finger_open = float(gripper.upper)
            self._finger_closed_command = float(gripper.lower)
            self._gripper_target = 1.0
            self._terminated = self._truncated = False
            self.coordinates = Coordinates.from_control(
                {
                    "coordinate_calibration": {
                        "base_xyz": self._base[:3, 3],
                        "base_rotation": self._base[:3, :3],
                        "table_top_xyz": [0.0, 0.0, 0.0],
                        "tcp": "native_grasp_tcp",
                    }
                }
            )
            self.tool_rotation = grasp_axes(np.array([0, 1, 0]), np.array([0, 0, 1]))
            self._state.save("coordinates.json", self.coordinates.metadata(), step=None)
        except BaseException:
            env.close()
            raise

    def get_task_language(self) -> str:
        """Expose the task-provided goal in the same frame as the action tools."""
        goal = self._obs["extra"]["goal_pos"][0].cpu().numpy()
        xyz = (self.coordinates.points(goal) * 100.0).round(2).tolist()
        return f"Move the red cube to the goal at general-frame xyz {xyz} cm."

    def native_success(self) -> bool:
        """Return the latched verdict from the native simulator."""
        return self._success

    def episode_flags(self) -> dict[str, bool]:
        """Preserve native termination and the RPent step cap."""
        return {
            "terminated": self._terminated,
            "truncated": self._truncated or self.env_steps >= self._max_episode_steps,
        }

    def _measure_native(self) -> Measurement:
        tcp = pose_matrix(self._agent.tcp_pose.raw_pose[0].cpu().numpy())
        gap = self._agent.robot.get_qpos()[0, -2:].sum().item()
        return Measurement(
            tcp[:3, 3], tcp[:3, :3], float(np.clip(gap / (2 * self._finger_open), 0, 1))
        )

    def _step(self, action: np.ndarray) -> None:
        self._check_cancelled()
        self._obs, _, terminated, truncated, info = self.env.step(action)
        self.env_steps += 1
        self._success |= bool(info["success"].item())
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)

    def _pose_action(
        self, current: Measurement, xyz: np.ndarray, rotation: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        root_rotation = self._base[:3, :3].T
        distance = float(np.linalg.norm(xyz - current.xyz))
        angle = Rotation.from_matrix(rotation @ current.rotation.T).magnitude()
        # The absolute controller consumes root-frame meters and intrinsic XYZ.
        action = np.array(
            [
                *(root_rotation @ (xyz - self._base[:3, 3])),
                *Rotation.from_matrix(root_rotation @ rotation).as_euler("XYZ"),
                self._gripper_target,
            ],
            dtype=np.float32,
        )
        return action, distance, angle

    def _gripper_action(self, opening: float, error: float, within: bool) -> float:
        finger = (
            opening * self._finger_open if opening > 0 else self._finger_closed_command
        )
        self._gripper_target = (
            2
            * (finger - self._finger_closed_command)
            / (self._finger_open - self._finger_closed_command)
            - 1
        )
        return self._gripper_target

    def robot_state(self) -> dict[str, Any]:
        """Report measured TCP and gripper state in RPent's general frame."""
        return self.measure().state()

    def _model_points(self, native: np.ndarray) -> np.ndarray:
        return self.coordinates.points(native)

    def capture(self, camera: str) -> tuple[np.ndarray, np.ndarray, dict]:
        """Project the native camera's metric depth into world coordinates."""
        sensor = self._obs["sensor_data"]["base_camera"]
        parameters = self._obs["sensor_param"]["base_camera"]
        rgb = sensor["rgb"][0].cpu().numpy()
        depth_m = sensor["depth"][0, ..., 0].cpu().numpy() / 1000.0
        world_to_camera = np.eye(4)
        world_to_camera[:3] = parameters["extrinsic_cv"][0, :3].cpu().numpy()
        calibration = {
            "intrinsic_K": parameters["intrinsic_cv"][0].cpu().tolist(),
            "extrinsic_cam2world": np.linalg.inv(world_to_camera).tolist(),
            "width": rgb.shape[1],
            "height": rgb.shape[0],
            "coordinates": self.coordinates.metadata(),
        }
        world = geometry.world_from_depth(depth_m, calibration)
        world[depth_m <= 0] = np.nan
        return rgb, world, calibration

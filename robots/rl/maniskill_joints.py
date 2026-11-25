# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Expose official ManiSkill joint-delta controllers for hands and locomotion."""

from __future__ import annotations

import numpy as np

from robots.rl.coordinates import Coordinates
from robots.rl.geometry import rpy_of
from robots.rl.maniskill_arm import transform
from rpent.tools import ToolResult
from rpent.utils.transforms import rotation_from_wxyz

JOINT_ROBOTS = {
    "allegro_hand_right_touch",
    "dclaw",
    "trifingerpro",
    "ant",
    "anymal_c",
    "cart_pole",
    "hopper",
    "humanoid",
    "unitree_g1_simplified_legs",
    "unitree_go2_simplified_locomotion",
    "unitree_h1_simplified",
}


class ManiSkillJointDrive:
    """Map each action channel from the active controller, omitting passive joints."""

    def __init__(self, agent):
        self.robot = agent.robot
        agent.set_control_mode("pd_joint_delta_pos")
        agent.controller.reset()
        ctrl = agent.controller
        self.last = np.zeros(ctrl.single_action_space.shape, dtype=np.float32)
        joints = self.robot.active_joints
        units = ["cm" if j.type[0] == "prismatic" else "rad" for j in joints]
        self.scales = np.array([100 if u == "cm" else 1 for u in units])
        channels, max_delta = [], []
        parts = getattr(ctrl, "controllers", {"joint": ctrl})
        for part in parts.values():
            if not part.single_action_space.shape[0]:
                continue
            space = part._original_single_action_space
            if (
                not part.config.use_delta
                or not part.config.normalize_action
                or len(part.joints) != space.shape[0]
                or not np.allclose(space.low, -space.high)
            ):
                raise ValueError(f"{agent.uid} requires symmetric joint-delta channels")
            channels.extend(j.name for j in part.joints)
            limits = space.high * self.scales[np.asarray(part.active_joint_indices)]
            max_delta.extend(round(float(limit), 6) for limit in limits)
        assert len(channels) == len(self.last)
        self.layout = {
            "mode": agent.control_mode,
            "action_order": channels,
            "max_delta_per_step": max_delta,
            "state_order": [j.name for j in joints],
            "state_units": units,
            "control_frequency_hz": agent.controller.control_freq,
        }
        base = transform(self.robot.pose)
        yaw = np.arctan2(base[1, 0], base[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        self.coordinates = Coordinates(
            base[:3, 3].copy(),
            np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
            tcp="native_joint_control",
        )

    def state(self, obs):
        base = transform(self.robot.pose)
        state = {
            "joint_position": (np.asarray(self.robot.qpos)[0] * self.scales)
            .round(5)
            .tolist(),
            "joint_velocity": (np.asarray(self.robot.qvel)[0] * self.scales)
            .round(5)
            .tolist(),
            "base_xyz": self.coordinates.points(base[:3, 3]).round(6).tolist(),
            "base_orientation_rpy": rpy_of(self.coordinates.orientation(base[:3, :3])),
        }

        # Only forward task signals already returned in the configured policy observation.
        extra = obs.get("extra", {})
        if "rotate_dir" in extra:
            direction = np.asarray(extra["rotate_dir"])
            state["rotate_dir"] = (
                direction @ self.coordinates.rotation
                if direction.shape == (3,)
                else direction
            ).tolist()
        if "valve_qpos" in extra:
            state["valve_angle_rad"] = float(np.asarray(extra["valve_qpos"]).item())
            state["valve_velocity_rad_s"] = float(
                np.asarray(extra["valve_qvel"]).item()
            )
        if "goal_q" in extra:
            world_rotation = rotation_from_wxyz(extra["goal_q"]).as_matrix()
            state["goal"] = {
                "xyz": self.coordinates.points(extra["goal_pos"]).round(6).tolist(),
                "orientation_rpy": rpy_of(self.coordinates.orientation(world_rotation)),
            }
        return state

    def drive(self, episode, values, steps):
        self.last[:] = values
        start = episode.env_steps
        for _ in range(steps):
            episode.step_controllers()
            if episode.ended:
                break
        count = episode.env_steps - start
        return ToolResult(data={"status": "executed", "env_steps": count})

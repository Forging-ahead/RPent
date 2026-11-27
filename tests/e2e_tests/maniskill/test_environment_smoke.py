# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Exercise the real ManiSkill controllers through RPent's episode entrypoint."""

import contextlib
import json
import math
import os

import numpy as np
import pytest

if os.environ.get("RPENT_RUN_MANISKILL_INTEGRATION") != "1":
    pytest.skip("Enable the native ManiSkill integration suite", allow_module_level=True)

import mani_skill.envs  # noqa: E402, F401
from mani_skill.utils.registration import REGISTERED_ENVS  # noqa: E402

from rlinf.agents.rpent.maniskill_tasks import CARTESIAN_TASKS  # noqa: E402
from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv  # noqa: E402
from robots.rl.geometry import rpy_of  # noqa: E402
from robots.rl.maniskill_arm import transform  # noqa: E402
from robots.rl.rl_episode import parse_args, start_environment  # noqa: E402
from rpent.utils.logging import init_output_dir  # noqa: E402


@pytest.fixture
def open_episode(tmp_path):
    init_output_dir(tmp_path)
    with contextlib.ExitStack() as release:
        def start(env_id):
            seed = 7 if env_id == "PickCube-v1" else 0
            task = {"environment": "maniskill", "task_name": env_id, "seed": seed}
            request = tmp_path / "request.json"
            request.write_text(json.dumps({"task": task}))
            args = parse_args(["--output-dir", str(tmp_path), "--request", str(request)])
            return start_environment(args, task, release)[0]

        yield start


@pytest.mark.parametrize("env_id", ["MS-CartpoleBalance-v1", "AnymalC-Reach-v1", "RotateValveLevel0-v1"])
def test_excluded_tasks_rejected_before_simulation(env_id, open_episode):
    with pytest.raises(ValueError, match="Unsupported ManiSkill pose task"):
        open_episode(env_id)


@pytest.mark.parametrize("env_id", sorted(CARTESIAN_TASKS))
def test_native_episode_caps_coordinates_and_horizon(env_id, open_episode, monkeypatch):
    monkeypatch.setattr(REGISTERED_ENVS[env_id], "max_episode_steps", 1)
    kit = open_episode(env_id)
    opening = kit.initial_observation().data
    caps, state = opening["sensors"], opening["state"]
    tools = {tool.name for tool in kit.list_tools()}
    assert ("view_env" in tools) == bool(caps["rgb_cameras"])
    assert ("back_project" in tools) == bool(caps["depth_cameras"])
    for camera in caps["rgb_cameras"]:
        assert kit.execute_tool("view_env", {"camera": camera}).images

    if env_id == "PickCube-v1":
        sim = kit._primitives
        assert isinstance(sim.env, ManiskillEnv) and sim.env.seed == 7
        assert sim._obs["sensor_data"]["base_camera"]["rgb"].shape == (512, 512, 3)
        goal = sim.env.unwrapped.goal_site.pose.p[0].cpu().numpy()
        assert state["goal"]["xyz"] == pytest.approx(100 * sim.coordinates.points(goal), abs=0.01)
        assert state["is_grasped"] is False and state["cube_size_cm"] == pytest.approx(4)
    if not caps["gripper"]:
        assert "opening" not in state
        close = {"actions": [{"action": "set_gripper", "opening": 1}]}
        assert kit.execute_tool("execute_actions", close).is_error

    selected = state["left"] if caps["dual_arm"] else state
    x, y, z = selected["pose"]["xyz"]
    move = {"action": "move_to", "xyz": [x, y, z + 2]}
    if caps["dual_arm"]:
        move["arm"] = "left"
    result = kit.execute_tool("execute_actions", {"actions": [move, move]}).data
    assert result["executed_count"] == 1
    metrics = kit.native_outcome()
    assert metrics["terminated"] or metrics["truncated"]
    assert metrics["success"] is None or type(metrics["success"]) is bool
    record = kit.state.latest_record().extras
    assert record["env_steps"] == 1 and math.isfinite(metrics["episode_return"])
    assert record["episode_return"] == pytest.approx(record["reward"])


@pytest.mark.parametrize("env_id", ["UnitreeG1PlaceAppleInBowl-v1", "UnitreeG1TransportBox-v1", "InsertFlower-v1"])
def test_cartesian_tools_on_native_joint_controllers(env_id, open_episode):
    kit = open_episode(env_id)
    state = kit.initial_observation().data["state"]
    sim = kit._primitives
    side = "left" if sim.dual_arm else "arm"
    arm = sim.arms[side]
    move = {"action": "move_to"}
    close = {"action": "set_gripper", "opening": 0.5}
    if sim.dual_arm:
        # Use an FK pose so the requested orientation is reachable on the five-axis arm.
        joints = np.asarray(arm.robot.get_qpos())[0].copy()
        elbow = arm.robot.active_joints_map["left_elbow_pitch_joint"]
        angle = -0.12 if "TransportBox" in env_id else 0.12
        joints[np.asarray(elbow.active_index).item()] += angle
        arm.pin.compute_forward_kinematics(joints)
        link = arm.pin.get_link_pose(np.asarray(arm.ee.index).item())
        pose = transform(arm.robot.root.pose) @ transform(link)
        rotation = sim.coordinates.orientation(pose[:3, :3] @ arm.tool_rotation)
        xyz = (100 * sim.coordinates.points(pose[:3, 3])).tolist()
        move.update(arm=side, xyz=xyz, orientation_rpy=rpy_of(rotation))
        close["arm"] = side
    else:
        x, y, z = state["pose"]["xyz"]
        move["xyz"] = [x, y, z + 2]
    result = kit.execute_tool("execute_actions", {"actions": [move, close]}).data
    assert [item["status"] for item in result["results"]] == ["reached", "reached"], result
    final = result["final_state"]
    measured = final[side] if sim.dual_arm else final
    assert measured["opening"] == pytest.approx(0.5, abs=0.08)
    if sim.dual_arm:
        assert final["right"]["pose"]["xyz"] == pytest.approx(state["right"]["pose"]["xyz"], abs=0.3)

# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Run one bounded RPent action call against each installed ManiSkill task."""

import contextlib
import json
import math
import os

import pytest

if os.environ.get("RPENT_RUN_MANISKILL_INTEGRATION") != "1":
    pytest.skip(
        "Enable the native ManiSkill integration suite", allow_module_level=True
    )

import mani_skill.envs  # noqa: E402, F401 - populates the native registry
from mani_skill.utils.registration import REGISTERED_ENVS  # noqa: E402

from robots.rl.maniskill_arm import UnsupportedRobotPolicy  # noqa: E402
from robots.rl.rl_episode import parse_args, start_environment  # noqa: E402
from rpent.utils.logging import init_output_dir  # noqa: E402


@pytest.fixture
def open_episode(tmp_path):
    """Start a real episode through the CLI contract and release it after the test."""
    init_output_dir(tmp_path)
    with contextlib.ExitStack() as release:

        def start(env_id, *, seed=0):
            task = {"environment": "maniskill", "task_name": env_id, "seed": seed}
            request = tmp_path / "request.json"
            request.write_text(json.dumps({"task": task}))
            argv = ["--output-dir", str(tmp_path), "--request", str(request)]
            toolkit, _ = start_environment(parse_args(argv), task, release)
            return toolkit

        yield start


def test_pick_cube_uses_rlinf_env():
    from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv

    from robots.rl.maniskill import _make_env

    env, task = _make_env("PickCube-v1", None, 1, seed=7)
    with contextlib.closing(env):
        assert task == "PickCube-v1"
        assert isinstance(env, ManiskillEnv)
        assert env.seed == 7
        obs, _ = env.reset()
        assert obs["sensor_data"]["base_camera"]["rgb"].shape == (512, 512, 3)
        _, reward, _, truncated, info = env.step(env.action_space.sample())
        assert bool(truncated)
        assert int(info["episode"]["episode_len"]) == 1
        assert float(info["episode"]["return"]) == pytest.approx(float(reward))


def test_pick_cube_reports_native_goal_in_the_planner_frame(open_episode):
    toolkit = open_episode("PickCube-v1", seed=7)
    observation = toolkit.initial_observation().data
    primitives = toolkit._primitives
    assert (
        primitives.env.cfg.init_params.max_episode_steps
        == REGISTERED_ENVS["PickCube-v1"].max_episode_steps
    )
    native_goal = primitives.env.unwrapped.goal_site.pose.p[0].cpu().numpy()
    expected_cm = 100 * primitives.coordinates.points(native_goal)
    assert observation["state"]["goal"]["xyz"] == pytest.approx(expected_cm, abs=0.01)
    assert observation["state"]["is_grasped"] is False
    assert observation["state"]["cube_size_cm"] == pytest.approx(4)


@pytest.mark.parametrize(
    "env_id",
    [
        "DrawTriangle-v1",
        "DrawSVG-v1",
        "FMBAssembly1Easy-v1",
        "UnitreeG1Stand-v1",
        "UnitreeH1Stand-v1",
    ],
)
def test_native_boolean_reward_environment(env_id):
    from robots.rl.maniskill import _make_env

    env, _ = _make_env(env_id, None, 2)
    with contextlib.closing(env):
        env.reset()
        for _ in range(2):
            _, reward, _, _, info = env.step(env.action_space.sample())
            assert reward.dtype.is_floating_point
            assert math.isfinite(float(info["episode"]["return"]))


@pytest.mark.parametrize("horizon", [None, 3])
def test_fetch_base_and_lift_use_effective_episode_budget(tmp_path, horizon):
    from robots.rl.maniskill import ManiSkillPrimitives
    from rpent.session import EnvState

    init_output_dir(tmp_path)
    sim = ManiSkillPrimitives(
        task_name="SceneManipulation-v1",
        seed=0,
        state=EnvState(tmp_path),
        check_cancelled=lambda: None,
        max_episode_steps=horizon,
    )
    with contextlib.closing(sim.env):
        before = sim.env_steps
        base = sim.move_base(vx=0.1, duration_s=0.6).data
        assert base["executed_steps"] > 0
        assert sim.env_steps > before
        if horizon is not None:
            assert sim.env_steps == horizon
            assert sim.truncated
        else:
            assert base["displacement_initial_base_m"][0] > 0.01
            lift = sim.move_lift(delta_m=-0.02, duration_s=1).data
            assert abs(lift["height_error_m"]) <= 0.004


@pytest.mark.parametrize(
    "env_id",
    ["UnitreeG1PlaceAppleInBowl-v1", "UnitreeG1TransportBox-v1", "InsertFlower-v1"],
)
def test_cartesian_tools_on_native_joint_controllers(env_id, open_episode):
    import numpy as np

    from robots.rl.geometry import rpy_of
    from robots.rl.maniskill_arm import transform

    kit = open_episode(env_id)
    opening = kit.initial_observation().data
    cameras = opening["sensors"]["rgb_cameras"]
    assert cameras
    for camera in cameras:
        assert len(kit.execute_tool("view_env", {"camera": camera}).images) == 1
    state = opening["state"]
    backend = kit._primitives
    for side, arm in backend.arms.items():
        selector = {"arm": side} if backend.dual_arm else {}
        if backend.dual_arm:
            # Each G1 arm has five joints. Generate a reachable full pose.
            robot = arm.robot
            qpos = np.asarray(robot.get_qpos())[0].copy()
            joint = robot.active_joints_map[f"{side}_elbow_pitch_joint"]
            angle = -0.12 if env_id == "UnitreeG1TransportBox-v1" else 0.12
            qpos[int(np.asarray(joint.active_index).item())] += angle
            arm.pin.compute_forward_kinematics(qpos)
            pose = (
                transform(robot.root.pose)
                @ arm.pin.get_link_pose(
                    int(np.asarray(arm.ee.index).item())
                ).to_transformation_matrix()
            )
            xyz = (100 * backend.coordinates.points(pose[:3, 3])).tolist()
            rpy = rpy_of(
                backend.coordinates.orientation(pose[:3, :3] @ arm.tool_rotation)
            )
            move = {"xyz": xyz, "orientation_rpy": rpy}
            other = "right" if side == "left" else "left"
            held = state[other]["pose"]["xyz"]
        else:
            x, y, z = state["pose"]["xyz"]
            move = {"xyz": [x, y, z + 2]}
        outcome = kit.execute_tool(
            "execute_actions",
            {
                "actions": [
                    {"action": "move_to", **selector, **move},
                    {"action": "set_gripper", **selector, "opening": 0.5},
                ]
            },
        ).data
        assert [r["status"] for r in outcome["results"]] == [
            "reached",
            "reached",
        ], outcome
        state = outcome["final_state"]
        measured = state[side] if backend.dual_arm else state
        assert measured["opening"] == pytest.approx(0.5, abs=0.08)
        if backend.dual_arm:
            np.testing.assert_allclose(state[other]["pose"]["xyz"], held, atol=0.3)


@pytest.mark.parametrize("env_id", sorted(REGISTERED_ENVS))
def test_native_environment_through_episode_entrypoint(
    env_id, open_episode, monkeypatch
):
    """Exercise the same tools and task routing used by RLinf's episode worker."""
    # A one-step registered horizon exercises native truncation inside this test only.
    monkeypatch.setattr(REGISTERED_ENVS[env_id], "max_episode_steps", 1)
    try:
        toolkit = open_episode(env_id)
    except UnsupportedRobotPolicy as exc:
        # Report unmapped embodiments separately from native installation or controller failures.
        if "no mapped arm" in str(exc):
            pytest.skip(str(exc))
        raise
    opening = toolkit.initial_observation().data
    caps = opening["sensors"]
    tools = {definition.name for definition in toolkit.list_tools()}
    assert ("view_env" in tools) == bool(caps["rgb_cameras"])
    assert ("back_project" in tools) == bool(caps["depth_cameras"])
    for camera in caps["rgb_cameras"]:
        response = toolkit.execute_tool("view_env", {"camera": camera})
        assert len(response.images) == 1, response.data

    state = opening["state"]
    if "joint_control" in caps:
        count = len(caps["joint_control"]["action_order"])
        move = {"action": "drive_joints", "values": [0.25] * count}
        invalid = {**move, "values": [0] * (count + 1)}
        rejected = toolkit.execute_tool("execute_actions", {"actions": [move, invalid]})
        assert "error" in rejected.data and toolkit._primitives.env_steps == 0
    else:
        pose = state["left"]["pose"] if caps["dual_arm"] else state["pose"]
        x, y, z = pose["xyz"]
        move = {"action": "move_to", "xyz": [x, y, z + 2]}
        if caps["dual_arm"]:
            move["arm"] = "left"
    actions = [move, move]
    outcome = toolkit.execute_tool("execute_actions", {"actions": actions})
    assert outcome.data.get("executed_count") == 1, outcome.data
    if "joint_control" in caps:
        assert outcome.data["final_state"]["joint_position"] != state["joint_position"]
    metrics = toolkit.native_outcome()
    assert metrics["terminated"] or metrics["truncated"]
    assert metrics["success"] is None or type(metrics["success"]) is bool
    assert math.isfinite(metrics["episode_return"])
    assert toolkit.state.latest_record().extras["env_steps"] == 1

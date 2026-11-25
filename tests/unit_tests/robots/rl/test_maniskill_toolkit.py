# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Public tool schemas, native batch execution and the durable episode result."""

import json
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from gymnasium.spaces import Box

from robots.rl import maniskill
from robots.rl.toolkit import RLToolkit, episode_tools
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.utils import templates
from tests.unit_tests.robots.rl.test_libero_rl_contracts import FakeSam3
from tests.unit_tests.robots.rl.test_maniskill_backend import NativeEpisode


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(templates, "default_variables", lambda: {})


def example_env(camera_mode="state", **episode_options):
    sensors, params = {}, {}
    if camera_mode in {"rgb", "rgbd"}:
        sensors["cam0"] = {"rgb": np.zeros((4, 5, 3), dtype=np.uint8)}
    if camera_mode == "rgbd":
        sensors["cam0"]["depth"] = np.full((4, 5, 1), 1000, dtype=np.int16)
        params["cam0"] = {"intrinsic_cv": np.eye(3), "extrinsic_cv": np.eye(4)[:3]}
    return NativeEpisode(
        {"sensor_data": sensors, "sensor_param": params},
        {"fail": False},
        **episode_options,
    )


def maniskill_toolkit(output_dir, monkeypatch, env, sam3=None, *, seed=0, **options):
    monkeypatch.setattr(maniskill, "_make_env", lambda *args: (env, "Example-v1"))
    return RLToolkit(
        runtime_kwargs={"task_name": "Example-v1", "seed": seed, "sam3_client": sam3},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(output_dir.parent / "memory"),
        output_dir=output_dir,
        environment="maniskill",
        **options,
    )


def execute(kit, *actions):
    return kit.execute_tool("execute_actions", {"actions": list(actions)})


@pytest.mark.parametrize(
    "mode,sam,perception",
    [
        ("state", True, set()),
        ("rgb", False, {"view_env"}),
        ("rgb", True, {"view_env", "segment"}),
        ("rgbd", False, {"view_env", "back_project"}),
    ],
)
def test_tools_match_policy_sensors_and_segment_service(
    tmp_path, monkeypatch, mode, sam, perception
):
    env = example_env(mode)
    client = FakeSam3(np.ones((4, 5), dtype=bool)) if sam else None
    toolkit = maniskill_toolkit(
        tmp_path / "episode", monkeypatch, env, client, image_size=256
    )
    with closing(toolkit) as kit:
        tools = {tool.name: tool for tool in kit.list_tools()}
        assert set(tools) == {"execute_actions"} | perception
        initial = kit.initial_observation().data
        assert initial["state"]["pose"]["xyz"] == [0, 0, 30]
        assert "action_space" not in initial
        schema = tools["execute_actions"].input_schema["properties"]["actions"]
        assert schema["minItems"] == 1 and schema["maxItems"] == 3
        libero = next(
            t for t in episode_tools(kit, "libero", 3) if t.name == "execute_actions"
        )
        assert tools["execute_actions"].input_schema == libero.input_schema
        if "view_env" in perception:
            view = kit.execute_tool("view_env", {"camera": "cam0"})
            assert len(view.images) == 1
            assert view.data["images"] == [{"camera": "cam0", "size": [256, 256]}]
        if "segment" in perception:
            mask = kit.execute_tool("segment", {"camera": "cam0", "prompt": "object"})
            assert mask.data["mask_id"] and mask.data["box"]
            assert mask.data["world_xyz"] is None
        if "back_project" in perception:
            point = kit.execute_tool("back_project", {"pixel": [0, 0], "step": 0})
            assert point.data["world_xyz"] == pytest.approx([-100, 0, 100])
            assert point.data["position_unit"] == "cm"
        assert not env.received


@pytest.mark.parametrize(
    "horizon,truncate,fail,reward",
    [(2, False, False, 2.0), (2, True, False, 2.0), (1, True, None, 0.5)],
)
def test_validate_entire_batch_and_stop_at_native_end(
    tmp_path, monkeypatch, horizon, truncate, fail, reward
):
    env = example_env(horizon=horizon, truncate=truncate, reward=reward)
    if fail is None:
        env.info.clear()
    move = {"action": "move_to", "xyz": [10, 0, 30]}
    with closing(maniskill_toolkit(tmp_path / "episode", monkeypatch, env)) as kit:
        assert not kit.tool_capabilities["rgb_cameras"]
        assert kit.native_outcome()["success"] is None
        for invalid in ([[0.5, -0.5], [2, 0]], [move, [2, 0]], [[0.5]], [move] * 4):
            assert execute(kit, *invalid).is_error
        assert not env.received
        result = execute(kit, move, {"action": "set_gripper", "opening": 0})
        assert result.data["executed_count"] == 1
        receipt = result.data["results"][0]
        assert receipt["terminated"] is not truncate
        assert receipt["truncated"] is truncate
        assert receipt["status"] == "partial"
        assert receipt["env_steps"] == horizon
        for receipt in [result.data, *result.data["results"]]:
            assert {"reward", "episode_return", "success", "fail"}.isdisjoint(receipt)
        np.testing.assert_allclose(
            [a[0, :3] for a in env.received], [[0.03, 0, 0.3], [0.06, 0, 0.3]][:horizon]
        )
        assert result.data["final_state"]["pose"]["xyz"] == [3.0 * horizon, 0.0, 30.0]
        expected = {
            "success": None,
            "fail": fail,
            "episode_return": reward * horizon,
            "terminated": not truncate,
            "truncated": truncate,
        }
        assert kit.native_outcome() == expected
        assert kit.completion_result() == {"status": "completed", **expected}
        assert kit.state.latest_step == 1
        record = kit.state.latest_record()
        assert record.terminated is not truncate and record.truncated is truncate
        assert record.extras["env_steps"] == horizon
        assert {key: record.extras[key] for key in expected} == expected
        with pytest.raises(RuntimeError, match="ended"):
            kit._primitives.step_controllers()
    assert env.closed
    rollout = json.loads((tmp_path / "episode" / "rollout.json").read_text())
    assert rollout["observations"] == 2 and rollout["env_steps"] == horizon
    assert {key: rollout[key] for key in expected} == expected


def test_drawing_arm_keeps_the_pose_tool_and_omits_gripper(tmp_path, monkeypatch):
    env = example_env()
    agent = env.unwrapped.agent
    agent.gripper_joint_names = []
    del agent.controller.controllers["gripper"]
    del agent.controller.action_mapping["gripper"]
    with closing(maniskill_toolkit(tmp_path / "episode", monkeypatch, env)) as kit:
        state = kit.initial_observation().data
        assert state["coordinates"]["tcp"] == "native_tool_tip"
        assert "opening" not in state["state"]
        assert execute(kit, {"action": "set_gripper", "opening": 1}).is_error
        assert not env.received
        result = execute(kit, {"action": "move_to", "xyz": [0, 0, 36]})
        assert result.data["executed_count"] == 1
        assert len(env.received) == 2
        assert result.data["final_state"]["pose"]["xyz"] == [0.0, 0.0, 36.0]


def test_gripper_target_persists_during_a_later_move(tmp_path, monkeypatch):
    env = example_env(horizon=None)
    with closing(maniskill_toolkit(tmp_path / "episode", monkeypatch, env)) as kit:
        result = execute(
            kit,
            {"action": "set_gripper", "opening": 0.25},
            {"action": "move_to", "xyz": [6, 0, 30]},
        )
        assert result.data["executed_count"] == 2
        assert all(item["status"] == "reached" for item in result.data["results"])
        assert result.data["final_state"]["opening"] == 0.25
        assert result.data["final_state"]["pose"]["xyz"] == [6.0, 0.0, 30.0]
        assert len(env.received) > 2
        np.testing.assert_allclose([a[0, 6] for a in env.received], -0.5, atol=1e-6)


def test_dual_arm_uses_native_labels_and_holds_idle_arm(tmp_path, monkeypatch):
    env = example_env(horizon=None)
    other = example_env()
    left, right = env.unwrapped.agent, other.unwrapped.agent
    for sign, agent in [(1, left), (-1, right)]:
        agent.robot.root.pose.p[1] = sign * 0.2
        agent.tcp.pose.p[1] = sign * 0.2
    env.unwrapped.left_agent, env.unwrapped.right_agent = left, right
    env.unwrapped.agent = SimpleNamespace(agents_dict={"second": right, "first": left})

    def apply_pair(actions):
        env.received.append(actions)
        left.tcp.pose.p[:] = left.robot.root.pose.p + actions["first"][0, :3]
        right.tcp.pose.p[:] = right.robot.root.pose.p + actions["second"][0, :3]

    env.apply = apply_pair
    with closing(maniskill_toolkit(tmp_path / "episode", monkeypatch, env)) as kit:
        assert kit.initial_observation().data["state"]["left"]["pose"]["xyz"] == [
            0.0,
            20.0,
            30.0,
        ]
        result = execute(
            kit, {"action": "move_to", "arm": "right", "xyz": [6, -20, 30]}
        )
        assert result.data["results"][0]["status"] == "reached"
        assert result.data["final_state"]["left"]["pose"]["xyz"] == [0.0, 20.0, 30.0]
        assert result.data["final_state"]["right"]["pose"]["xyz"] == [6.0, -20.0, 30.0]


def test_fetch_uses_mobile_batch_and_validates_lateral_velocity():
    caps = {
        "rgb_cameras": [],
        "depth_cameras": [],
        "segment": False,
        "dual_arm": False,
        "gripper": True,
        "mobile": True,
    }
    fetch = next(
        t
        for t in episode_tools(None, "maniskill", 4, capabilities=caps)
        if t.name == "execute_actions"
    )
    base = {"action": "move_base", "vx": 10, "vy": 0, "yaw_rate": 0.2, "duration_s": 1}
    fetch.validate_arguments(
        {"actions": [base, {"action": "move_lift", "delta_cm": 2}]}
    )
    with pytest.raises(ValueError):
        fetch.validate_arguments({"actions": [{**base, "vy": 10}]})
    robocasa = next(
        t for t in episode_tools(None, "robocasa", 4) if t.name == "execute_actions"
    )
    robocasa.validate_arguments({"actions": [{**base, "vy": 10}]})


def test_fetch_controller_converts_drive_and_torso_to_native_inputs():
    def component(low, high):
        return SimpleNamespace(
            _original_single_action_space=Box(
                np.array(low, np.float32), np.array(high, np.float32)
            )
        )

    arm = SimpleNamespace(
        last=np.array([0.7, 0, 0, 0, 0, 0], dtype=float),
        slices={"base": slice(1, 3), "body": slice(3, 6)},
        parts={
            "base": component([-1, -3.14], [1, 3.14]),
            "body": component([-0.1] * 3, [0.1] * 3),
        },
    )
    sim = SimpleNamespace(arms={"arm": arm}, _lift_index=2, step_controllers=Mock())
    maniskill.ManiSkillPrimitives._mobile_step(sim, np.array([0.2, 0, 0.314]), 0.025)
    np.testing.assert_allclose(arm.last, [0.7, 0.2, 0.1, 0, 0, 0.25], atol=1e-6)
    maniskill.ManiSkillPrimitives._mobile_step(sim, check_cancelled=False)
    np.testing.assert_allclose(arm.last, [0.7, 0, 0, 0, 0, 0], atol=1e-6)
    sim.step_controllers.assert_called_with(check_cancelled=False)

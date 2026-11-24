# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Public tool schemas, native batch execution and the durable episode result."""

import json

import numpy as np
import pytest

from robots.rl import maniskill
from robots.rl.toolkit import RLToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.utils import templates
from tests.unit_tests.robots.rl.test_libero_rl_contracts import FakeSam3
from tests.unit_tests.robots.rl.test_maniskill_backend import NativeEpisode


class ExampleManiSkillEnv(NativeEpisode):
    """Two native control steps with a configurable policy observation."""

    def __init__(self, *, camera_mode="rgbd", truncate=False):
        sensors, params = {}, {}
        if camera_mode in {"rgb", "rgbd"}:
            sensors["cam0"] = {"rgb": np.zeros((4, 5, 3), dtype=np.uint8)}
        if camera_mode == "rgbd":
            sensors["cam0"]["depth"] = np.full((4, 5, 1), 1000, dtype=np.int16)
            params["cam0"] = {"intrinsic_cv": np.eye(3), "extrinsic_cv": np.eye(4)[:3]}
        super().__init__({"sensor_data": sensors, "sensor_param": params}, {})
        self.truncate = truncate
        self.closed = False

    def reset(self, *, seed):
        self.received.clear()
        self.obs["agent"] = {"qpos": [0, 0]}
        return self.obs, {}

    def step(self, action):
        self.apply(action)
        self.obs["agent"] = {"qpos": action.tolist()}
        count = len(self.received)
        info = {"fail": False, "episode": {"return": 2.0 * count, "episode_len": count}}
        return (
            self.obs,
            2.0,
            count == 2 and not self.truncate,
            count == 2 and self.truncate,
            info,
        )

    def close(self):
        self.closed = True


def make_toolkit(tmp_path, monkeypatch, env, sam3=None):
    monkeypatch.setattr(maniskill, "_make_env", lambda *args: (env, "Example-v1"))
    monkeypatch.setattr(templates, "default_variables", lambda: {})
    return RLToolkit(
        runtime_kwargs={"task_name": "Example-v1", "seed": 0, "sam3_client": sam3},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(tmp_path / "memory"),
        output_dir=tmp_path / "episode",
        image_size=256,
        environment="maniskill",
    )


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
    env = ExampleManiSkillEnv(camera_mode=mode)
    client = FakeSam3(np.ones((4, 5), dtype=bool)) if sam else None
    kit = make_toolkit(tmp_path, monkeypatch, env, client)
    try:
        tools = {tool.name: tool for tool in kit.list_tools()}
        assert set(tools) == {"execute_actions"} | perception
        initial = kit.initial_observation().data
        assert initial["state"]["pose"]["xyz"] == [0, 0, 30]
        assert "action_space" not in initial
        schema = tools["execute_actions"].input_schema["properties"]["actions"]
        assert schema["minItems"] == 1 and schema["maxItems"] == 3
        from robots.rl.toolkit import episode_tools

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
    finally:
        kit.close()


@pytest.mark.parametrize("truncate", [False, True])
def test_validate_entire_batch_and_stop_at_native_end(tmp_path, monkeypatch, truncate):
    env = ExampleManiSkillEnv(camera_mode="state", truncate=truncate)
    kit = make_toolkit(tmp_path, monkeypatch, env)
    try:
        for invalid in ([[0.5, -0.5], [2, 0]], [[0.5]], [[0, 0]] * 4):
            assert kit.execute_tool("execute_actions", {"actions": invalid}).is_error
        assert not env.received
        result = kit.execute_tool(
            "execute_actions",
            {
                "actions": [
                    {"action": "move_to", "xyz": [10, 0, 30]},
                    {"action": "set_gripper", "opening": 0},
                ]
            },
        )
        assert result.data["executed_count"] == 1
        receipt = result.data["results"][0]
        assert receipt["terminated"] is not truncate
        assert receipt["truncated"] is truncate
        assert receipt["status"] == "partial"
        assert receipt["env_steps"] == 2
        for receipt in [result.data, *result.data["results"]]:
            assert {"reward", "episode_return", "success", "fail"}.isdisjoint(receipt)
        np.testing.assert_allclose([a[0, 0] for a in env.received], [0.03, 0.06])
        assert result.data["final_state"]["pose"]["xyz"] == [6.0, 0.0, 30.0]
        expected = {
            "success": None,
            "fail": False,
            "episode_return": 4.0,
            "terminated": not truncate,
            "truncated": truncate,
        }
        assert kit.native_outcome() == expected
        assert kit.completion_result() == {"status": "completed", **expected}
        assert kit.state.latest_step == 1
        extras = kit.state.latest_record().extras
        assert {key: extras[key] for key in expected} == expected
    finally:
        kit.close()
    assert env.closed
    rollout = json.loads((tmp_path / "episode" / "rollout.json").read_text())
    assert rollout["observations"] == 2 and rollout["env_steps"] == 2
    assert {key: rollout[key] for key in expected} == expected


def test_drawing_arm_keeps_the_pose_tool_and_omits_gripper(tmp_path, monkeypatch):
    env = ExampleManiSkillEnv(camera_mode="state")
    agent = env.unwrapped.agent
    agent.gripper_joint_names = []
    del agent.controller.controllers["gripper"]
    del agent.controller.action_mapping["gripper"]
    kit = make_toolkit(tmp_path, monkeypatch, env)
    try:
        state = kit.initial_observation().data
        assert state["coordinates"]["tcp"] == "native_tool_tip"
        assert "opening" not in state["state"]
        assert kit.execute_tool(
            "execute_actions", {"actions": [{"action": "set_gripper", "opening": 1}]}
        ).is_error
        assert not env.received
        result = kit.execute_tool(
            "execute_actions", {"actions": [{"action": "move_to", "xyz": [0, 0, 36]}]}
        )
        assert result.data["executed_count"] == 1
        assert len(env.received) == 2
        assert result.data["final_state"]["pose"]["xyz"] == [0.0, 0.0, 36.0]
    finally:
        kit.close()


def test_gripper_target_persists_during_a_later_move(tmp_path, monkeypatch):
    env = ExampleManiSkillEnv(camera_mode="state")

    def step(action):
        env.apply(action)
        return (
            env.obs,
            0.0,
            False,
            False,
            {
                "success": False,
                "episode": {"return": 0.0, "episode_len": len(env.received)},
            },
        )

    env.step = step
    kit = make_toolkit(tmp_path, monkeypatch, env)
    try:
        result = kit.execute_tool(
            "execute_actions",
            {
                "actions": [
                    {"action": "set_gripper", "opening": 0.25},
                    {"action": "move_to", "xyz": [6, 0, 30]},
                ]
            },
        )
        assert result.data["executed_count"] == 2
        assert all(item["status"] == "reached" for item in result.data["results"])
        assert result.data["final_state"]["opening"] == 0.25
        assert result.data["final_state"]["pose"]["xyz"] == [6.0, 0.0, 30.0]
        assert len(env.received) > 2
        np.testing.assert_allclose([a[0, 6] for a in env.received], -0.5, atol=1e-6)
    finally:
        kit.close()


def test_dual_arm_uses_native_labels_and_holds_idle_arm(tmp_path, monkeypatch):
    env = ExampleManiSkillEnv(camera_mode="state")
    other = ExampleManiSkillEnv(camera_mode="state")
    left, right = env.unwrapped.agent, other.unwrapped.agent
    for sign, agent in [(1, left), (-1, right)]:
        agent.robot.root.pose.p[1] = sign * 0.2
        agent.tcp.pose.p[1] = sign * 0.2
    env.unwrapped.left_agent, env.unwrapped.right_agent = left, right
    from types import SimpleNamespace

    env.unwrapped.agent = SimpleNamespace(agents_dict={"second": right, "first": left})

    def step(actions):
        env.received.append(actions)
        left.tcp.pose.p[:] = left.robot.root.pose.p + actions["first"][0, :3]
        right.tcp.pose.p[:] = right.robot.root.pose.p + actions["second"][0, :3]
        return (
            env.obs,
            0.0,
            False,
            False,
            {"episode": {"return": 0.0, "episode_len": len(env.received)}},
        )

    env.step = step
    kit = make_toolkit(tmp_path, monkeypatch, env)
    try:
        assert kit.initial_observation().data["state"]["left"]["pose"]["xyz"] == [
            0.0,
            20.0,
            30.0,
        ]
        result = kit.execute_tool(
            "execute_actions",
            {"actions": [{"action": "move_to", "arm": "right", "xyz": [6, -20, 30]}]},
        )
        assert result.data["results"][0]["status"] == "reached"
        assert result.data["final_state"]["left"]["pose"]["xyz"] == [0.0, 20.0, 30.0]
        assert result.data["final_state"]["right"]["pose"]["xyz"] == [6.0, -20.0, 30.0]
    finally:
        kit.close()

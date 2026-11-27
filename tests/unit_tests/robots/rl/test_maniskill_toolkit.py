# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.

import json
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robots.rl.maniskill import ManiSkillPrimitives
from robots.rl.toolkit import RLToolkit, episode_tools
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.utils import templates
from tests.unit_tests.robots.rl.test_libero_rl_contracts import FakeSam3
from tests.unit_tests.robots.rl.test_maniskill_backend import NativeEpisode, stub_environment


def example_env(camera_mode="state", **options):
    sensors, params = {}, {}
    if camera_mode != "state":
        sensors["cam0"] = {"rgb": np.zeros((4, 5, 3), dtype=np.uint8)}
    if camera_mode == "rgbd":
        sensors["cam0"]["depth"] = np.full((4, 5, 1), 1000)
        params["cam0"] = {"intrinsic_cv": np.eye(3), "extrinsic_cv": np.eye(4)[:3]}
    return NativeEpisode({"sensor_data": sensors, "sensor_param": params}, {}, **options)


def maniskill_toolkit(output_dir, monkeypatch, env, sam3=None, *, seed=0, **options):
    monkeypatch.setattr(templates, "default_variables", lambda: {})
    stub_environment(monkeypatch, env, "Example-v1")
    return RLToolkit(
        runtime_kwargs={"task_name": "Example-v1", "seed": seed, "sam3_client": sam3},
        dashboard_events=NullDashboardEventSink(),
        memory=MemoryManager(output_dir.parent / "memory"),
        output_dir=output_dir,
        environment="maniskill",
        **options,
    )


@pytest.mark.parametrize("mode,sam,views", [
    ("state", False, set()), ("rgb", False, {"view_env"}),
    ("rgb", True, {"view_env", "segment"}), ("rgbd", False, {"view_env", "back_project"}),
])
def test_camera_tools_and_shared_action_contract(tmp_path, monkeypatch, mode, sam, views):
    client = FakeSam3(np.ones((4, 5), dtype=bool)) if sam else None
    toolkit = maniskill_toolkit(tmp_path / "episode", monkeypatch, example_env(mode), client)
    with closing(toolkit) as kit:
        tools = {tool.name: tool for tool in kit.list_tools()}
        assert set(tools) == {"execute_actions"} | views
        libero_action = next(
            tool for tool in episode_tools(kit, "libero", 3) if tool.name == "execute_actions"
        )
        assert tools["execute_actions"].input_schema == libero_action.input_schema
        state = kit.initial_observation().data["state"]
        assert state["pose"]["xyz"] == [0, 0, 30]
        if sam:
            mask = kit.execute_tool("segment", {"camera": "cam0", "prompt": "object"}).data
            assert mask["mask_id"] and mask["world_xyz"] is None
        if mode == "rgbd":
            point = kit.execute_tool("back_project", {"pixel": [0, 0], "step": 0}).data
            assert point["world_xyz"] == [-100, 0, 100] and point["position_unit"] == "cm"


@pytest.mark.parametrize("truncate", [False, True])
def test_batch_validation_and_native_outcome(tmp_path, monkeypatch, truncate):
    env = example_env(truncate=truncate)
    move = {"action": "move_to", "xyz": [10, 0, 30]}
    output = tmp_path / "episode"
    with closing(maniskill_toolkit(output, monkeypatch, env)) as kit:
        def execute(actions):
            return kit.execute_tool("execute_actions", {"actions": actions})

        assert execute([move, [2, 0]]).is_error and not env.received
        batch = execute([move, {"action": "set_gripper", "opening": 0}]).data
        assert batch["executed_count"] == 1 and len(env.received) == 2
        assert batch["results"][0]["status"] == "partial"
        assert {"reward", "episode_return", "success", "fail"}.isdisjoint(batch["results"][0])
        expected = {
            "success": None, "fail": None, "episode_return": 4.0,
            "terminated": not truncate, "truncated": truncate,
        }
        assert kit.completion_result() == {"status": "completed", **expected}
        with pytest.raises(RuntimeError, match="ended"):
            kit._primitives.step_controllers()
    rollout = json.loads((output / "rollout.json").read_text())
    assert env.closed and {key: rollout[key] for key in expected} == expected


def test_fetch_schema_and_native_drive_conversion():
    caps = dict(
        rgb_cameras=[], depth_cameras=[], segment=False, dual_arm=False, gripper=True, mobile=True
    )
    fetch = next(episode_tools(None, "maniskill", 4, capabilities=caps))
    base = {"action": "move_base", "vx": 10, "vy": 0, "yaw_rate": 0.2, "duration_s": 1}
    fetch.validate_arguments({"actions": [base, {"action": "move_lift", "delta_cm": 2}]})
    with pytest.raises(ValueError):
        fetch.validate_arguments({"actions": [{**base, "vy": 10}]})

    def controller(low, high):
        space = SimpleNamespace(low=np.array(low), high=np.array(high), shape=(len(low),))
        return SimpleNamespace(_original_single_action_space=space)

    arm = SimpleNamespace(
        last=np.array([0.7, 0, 0, 0, 0, 0]),
        slices={"base": slice(1, 3), "body": slice(3, 6)},
        parts={"base": controller([-1, -3.14], [1, 3.14]), "body": controller([-0.1] * 3, [0.1] * 3)},
    )
    sim = SimpleNamespace(arms={"arm": arm}, _lift_index=2, step_controllers=Mock())
    ManiSkillPrimitives._mobile_step(sim, np.array([0.2, 0, 0.314]), 0.025)
    np.testing.assert_allclose(arm.last, [0.7, 0.2, 0.1, 0, 0, 0.25], atol=1e-6)
    ManiSkillPrimitives._mobile_step(sim, check_cancelled=False)
    np.testing.assert_allclose(arm.last, [0.7, 0, 0, 0, 0, 0], atol=1e-6)
    sim.step_controllers.assert_called_with(check_cancelled=False)

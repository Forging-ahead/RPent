# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""ManiSkill episode contracts. Set RPENT_RUN_MANISKILL_INTEGRATION=1 for native simulation."""

import argparse
import asyncio
import json
import math
import os
import threading
from contextlib import ExitStack, closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robots.rl import maniskill, remote, rl_episode
from robots.rl.geometry import rpy_of
from robots.rl.maniskill import ManiSkillPrimitives
from robots.rl.maniskill_arm import ManiSkillArm, UnsupportedRobotPolicy, transform
from robots.rl.remote_manager import validate_request
from robots.rl.robot_spec import get_robot_spec
from robots.rl.toolkit import RLToolkit, episode_tools
from robots.rl.tools import Measurement
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.utils import templates
from rpent.utils.logging import init_output_dir
from rpent.utils.rpc.http_rpc import HttpRpcServer
from tests.unit_tests.robots.rl.test_libero_rl_contracts import FakeSam3
from tests.unit_tests.robots.rl.test_remote_contracts import Bridge, _owned_request


class ArmDouble(ManiSkillArm):
    """Keep the shared servo and stub the simulator-specific actuator conversion."""

    def __init__(self, episode, agent):
        self.episode, self.agent = episode, agent
        self._config, self.mount = episode._config, agent.mount
        self.has_gripper, self.tool_rotation = agent.has_gripper, np.eye(3)
        self.opening_target, self.last = 1.0, agent.state.copy()

    def _measure_native(self):
        return Measurement(self.agent.state[:3].copy(), np.eye(3), self.agent.state[3])

    def _pose_action(self, current, xyz, rotation):
        delta, _, distance, angle = self._bounded_pose_step(current, xyz, rotation)
        return np.r_[current.xyz + delta, self.opening_target], distance, angle

    def _command_gripper(self, action, opening, error, within):
        self.opening_target = action[3] = opening


class NativeEpisode:
    instruction = ""

    def __init__(self, obs, info, *, horizon=2, truncate=False):
        pose = np.eye(4)
        pose[0, 3] = 1
        mount = SimpleNamespace(
            pose=SimpleNamespace(to_transformation_matrix=lambda: pose)
        )
        agent = SimpleNamespace(
            uid="example_arm",
            mount=mount,
            has_gripper=True,
            state=np.array([1, 0, 0.3, 1.0]),
        )
        self.unwrapped = SimpleNamespace(agent=agent)
        self.obs, self.info = obs, info
        self.horizon, self.truncate = horizon, truncate
        self.received = []
        self.closed = False

    def reset(self, *, seed):
        self.seed = seed
        return self.obs, self.info

    def step(self, action):
        self.received.append(action)
        self.unwrapped.agent.state[:] = action[0]
        count = len(self.received)
        end = count == self.horizon
        info = {**self.info, "episode": {"return": 2.0 * count, "episode_len": count}}
        return self.obs, 2.0, end and not self.truncate, end and self.truncate, info

    def close(self):
        self.closed = True


def test_recorded_camera_calibration_and_video_frames(make_kit):
    rgb = np.full((2, 3, 3), 92, dtype=np.uint8)
    depth = np.array([[0, 1000, 2000]] * 2)
    extrinsic = np.eye(4)[:3]
    extrinsic[0, 3] = -2
    obs = {
        "sensor_data": {"head": {"rgb": rgb}, "side": {"rgb": rgb, "depth": depth}},
        "sensor_param": {
            "side": {"extrinsic_cv": extrinsic, "intrinsic_cv": np.eye(3)}
        },
    }
    env = NativeEpisode(obs, {"success": False})
    with make_kit(env, seed=5) as kit:
        backend, state = kit._primitives, kit.state
        record = state.latest_record()
        assert env.seed == 5 and backend.depth_cameras == ("side",)
        assert record.extras["success"] is False and record.extras["fail"] is None
        assert (
            "head.png" in record.artifacts and "head_world.npz" not in record.artifacts
        )
        world = state.load("side_world.npz", step=record.step_idx)
        np.testing.assert_allclose(world[0, 1], [3, 0, 1])
        assert np.isnan(world[0, 0]).all()
        rgb[:] = 93
        backend.step_controllers()
        assert [int(frame[0, 0, 0]) for frame in backend.frames] == [92, 93]


def test_unsupported_robot_does_not_get_a_fabricated_arm():
    episode = SimpleNamespace(_config=None, env_id="Native-v1")
    agent = SimpleNamespace(robot=object(), uid="unmapped")
    with pytest.raises(UnsupportedRobotPolicy, match="no mapped arm"):
        ManiSkillArm(episode, agent)


def example_env(camera_mode="state", **options):
    sensors, params = {}, {}
    if camera_mode != "state":
        sensors["cam0"] = {"rgb": np.zeros((4, 5, 3), dtype=np.uint8)}
    if camera_mode == "rgbd":
        sensors["cam0"]["depth"] = np.full((4, 5, 1), 1000)
        params["cam0"] = {"intrinsic_cv": np.eye(3), "extrinsic_cv": np.eye(4)[:3]}
    return NativeEpisode(
        {"sensor_data": sensors, "sensor_param": params}, {}, **options
    )


@pytest.fixture
def make_kit(tmp_path, monkeypatch):
    monkeypatch.setattr(templates, "default_variables", lambda: {})
    monkeypatch.setattr(maniskill, "ManiSkillArm", ArmDouble)

    def make(env, sam3=None, *, seed=0):
        monkeypatch.setattr(maniskill, "_make_env", lambda *args: (env, "Example-v1"))
        kit = RLToolkit(
            runtime_kwargs={
                "task_name": "Example-v1",
                "seed": seed,
                "sam3_client": sam3,
            },
            dashboard_events=NullDashboardEventSink(),
            memory=MemoryManager(tmp_path / "memory"),
            output_dir=tmp_path / "episode",
            environment="maniskill",
        )
        return closing(kit)

    return make


@pytest.mark.parametrize(
    "mode,sam,views",
    [
        ("state", False, set()),
        ("rgb", False, {"view_env"}),
        ("rgb", True, {"view_env", "segment"}),
        ("rgbd", False, {"view_env", "back_project"}),
    ],
)
def test_camera_tools_and_shared_action_contract(make_kit, mode, sam, views):
    client = FakeSam3(np.ones((4, 5), dtype=bool)) if sam else None
    with make_kit(example_env(mode), client) as kit:
        tools = {tool.name: tool for tool in kit.list_tools()}
        assert set(tools) == {"execute_actions"} | views
        libero_action = next(
            tool
            for tool in episode_tools(kit, "libero", 3)
            if tool.name == "execute_actions"
        )
        assert tools["execute_actions"].input_schema == libero_action.input_schema
        state = kit.initial_observation().data["state"]
        assert state["pose"]["xyz"] == [0, 0, 30]
        if sam:
            mask = kit.execute_tool(
                "segment", {"camera": "cam0", "prompt": "object"}
            ).data
            assert mask["mask_id"] and mask["world_xyz"] is None
        if mode == "rgbd":
            point = kit.execute_tool("back_project", {"pixel": [0, 0], "step": 0}).data
            assert (
                point["world_xyz"] == [-100, 0, 100] and point["position_unit"] == "cm"
            )


@pytest.mark.parametrize("truncate", [False, True])
def test_batch_validation_and_native_outcome(make_kit, truncate):
    env = example_env(truncate=truncate)
    move = {"action": "move_to", "xyz": [10, 0, 30]}
    with make_kit(env) as kit:

        def execute(actions):
            return kit.execute_tool("execute_actions", {"actions": actions})

        assert execute([move, [2, 0]]).is_error and not env.received
        batch = execute([move, {"action": "set_gripper", "opening": 0}]).data
        assert batch["executed_count"] == 1 and len(env.received) == 2
        assert batch["results"][0]["status"] == "partial"
        assert {"reward", "episode_return", "success", "fail"}.isdisjoint(
            batch["results"][0]
        )
        expected = {
            "success": None,
            "fail": None,
            "episode_return": 4.0,
            "terminated": not truncate,
            "truncated": truncate,
        }
        assert kit.completion_result() == {"status": "completed", **expected}
        with pytest.raises(RuntimeError, match="ended"):
            kit._primitives.step_controllers()
    rollout = json.loads((kit.output_dir / "rollout.json").read_text())
    assert env.closed and {key: rollout[key] for key in expected} == expected


def test_fetch_schema_and_native_drive_conversion():
    caps = {
        "rgb_cameras": [],
        "depth_cameras": [],
        "segment": False,
        "dual_arm": False,
        "gripper": True,
        "mobile": True,
    }
    fetch = next(episode_tools(None, "maniskill", 4, capabilities=caps))
    base = {"action": "move_base", "vx": 10, "vy": 0, "yaw_rate": 0.2, "duration_s": 1}
    fetch.validate_arguments(
        {"actions": [base, {"action": "move_lift", "delta_cm": 2}]}
    )
    with pytest.raises(ValueError):
        fetch.validate_arguments({"actions": [{**base, "vy": 10}]})

    def controller(low, high):
        space = SimpleNamespace(
            low=np.array(low), high=np.array(high), shape=(len(low),)
        )
        return SimpleNamespace(_original_single_action_space=space)

    arm = SimpleNamespace(
        last=np.array([0.7, 0, 0, 0, 0, 0]),
        slices={"base": slice(1, 3), "body": slice(3, 6)},
        parts={
            "base": controller([-1, -3.14], [1, 3.14]),
            "body": controller([-0.1] * 3, [0.1] * 3),
        },
    )
    sim = SimpleNamespace(arms={"arm": arm}, _lift_index=2, step_controllers=Mock())
    ManiSkillPrimitives._mobile_step(sim, np.array([0.2, 0, 0.314]), 0.025)
    np.testing.assert_allclose(arm.last, [0.7, 0.2, 0.1, 0, 0, 0.25], atol=1e-6)
    ManiSkillPrimitives._mobile_step(sim, check_cancelled=False)
    np.testing.assert_allclose(arm.last, [0.7, 0, 0, 0, 0, 0], atol=1e-6)
    sim.step_controllers.assert_called_with(check_cancelled=False)


def test_model_retry_uses_declared_robot_action_contract(make_kit):
    import time

    from pydantic_ai import ModelRetry, ToolCallPart

    from robots.rl.workflow import WorkflowCapability

    with make_kit(example_env()) as kit:
        workflow = WorkflowCapability(kit)
        workflow.request_started = time.perf_counter()
        request = SimpleNamespace(model=SimpleNamespace())
        call = ToolCallPart("execute_actions", {"actions": [{"action": "invalid"}]})
        response = SimpleNamespace(
            state="complete", finish_reason="tool_call", parts=[call]
        )
        with pytest.raises(ModelRetry, match="move_to.*set_gripper"):
            asyncio.run(
                workflow.after_model_request(
                    None, request_context=request, response=response
                )
            )
        assert workflow.stats["invalid_tool_responses"] == 1


@pytest.mark.parametrize(
    ("terminated", "truncated", "fail", "reason"),
    [
        (True, False, True, "environment_failure"),
        (True, False, None, "environment_terminated"),
        (False, True, None, "environment_step_limit"),
        (False, False, None, "request_limit"),
    ],
)
def test_maniskill_outcome_preserves_native_return_without_success(
    terminated, truncated, fail, reason
):
    native = {
        "success": None,
        "fail": fail,
        "episode_return": 2.75,
        "terminated": terminated,
        "truncated": truncated,
    }
    toolkit = SimpleNamespace(
        environment="maniskill",
        native_outcome=lambda: native,
        state=SimpleNamespace(records=lambda: []),
    )
    result = SimpleNamespace(stats={"turns_used": 4}, error=None, error_kind=None)
    outcome = rl_episode.episode_outcome(result, toolkit, max_turns=4)
    assert outcome == {
        **native,
        "terminal": True,
        "termination_reason": reason,
        "error": None,
        "error_kind": None,
    }
    toolkit.infrastructure_error = "RPC disconnected"
    failed = rl_episode.episode_outcome(result, toolkit, max_turns=4)
    assert (
        failed["termination_reason"] == "infrastructure_error"
        and not failed["terminal"]
    )
    assert {key: failed[key] for key in native} == native


@pytest.mark.parametrize("name", ["MS-AntRun-v1", "pick_cube"])
def test_local_maniskill_config_preserves_task_seed_and_native_step_budget(
    tmp_path, name
):
    parser = argparse.ArgumentParser()
    spec = get_robot_spec()
    spec.add_cli_args(parser, False)
    args = parser.parse_args(
        f"--simulator maniskill --task-name {name} --seed 19 --max-episode-steps 12 --image_size 256 --cuda-device 0".split()
    )
    args.output_dir = tmp_path
    daemons, kwargs = spec.init_runtime(
        args, tmp_path, NullDashboardEventSink(), {"env", "sam3"}
    )
    assert daemons == []
    assert kwargs == {
        "task_name": name,
        "seed": 19,
        "render_device": args.cuda_device,
        "sam3_client": None,
    }
    args.remote_toolkit_endpoint = "http://localhost:28490"
    _, runtime = spec.init_runtime(
        args, tmp_path, NullDashboardEventSink(), {"env", "sam3"}
    )
    remote_request = runtime["remote_request"]
    validate_request(remote_request, {0})
    assert remote_request["max_episode_steps"] == 12
    assert (
        remote_request["task"]["task_name"] == name
        and remote_request["task"]["seed"] == 19
    )


def test_maniskill_remote_request_rejects_non_native_or_path_identifiers():
    request = _owned_request()
    request["task"] = {"environment": "maniskill", "seed": 0}
    for name in ("../PickCube-v1", "NotVersioned", "a/b-v1", ""):
        with pytest.raises(ValueError, match="invalid native task identifier"):
            validate_request(
                {**request, "task": {**request["task"], "task_name": name}}, {0}
            )


@pytest.fixture
def connected(tmp_path, monkeypatch, request, make_kit):
    boot_id = Path("/proc/sys/kernel/random/boot_id")
    if not boot_id.exists():
        read_text = Path.read_text
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda path, *a, **kw: (
                "test-worker" if path == boot_id else read_text(path, *a, **kw)
            ),
        )
    with ExitStack() as release:
        kit = release.enter_context(make_kit(example_env(request.param), seed=1))
        task = {"environment": "maniskill", "task_name": "Example-v1", "seed": 1}
        runtime = {"task": task, "gpu": 0, "max_episode_steps": 10000}
        worker = remote.ToolkitWorker(
            kit, release, {**runtime, "output_dir": str(kit.output_dir)}
        )
        bridge = Bridge(worker)
        with HttpRpcServer(("127.0.0.1", 0), bridge._dispatch) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with closing(
                    remote.RemoteToolkit(
                        endpoint=f"http://127.0.0.1:{server.server_port}",
                        request=runtime,
                        config=SimpleNamespace(output_dir=tmp_path / "frontend"),
                        dashboard_events=NullDashboardEventSink(),
                        memory=MemoryManager(tmp_path / "memory"),
                    )
                ) as client:
                    yield client, kit, worker
                assert bridge.released
            finally:
                server.shutdown()
                thread.join()


@pytest.mark.parametrize("connected", ["rgbd", "state"], indirect=True)
def test_remote_maniskill_negotiates_native_action_and_camera_tools(connected):
    client, kit, worker = connected
    assert remote.declarations(client) == remote.declarations(kit)
    assert client.native_outcome() == kit.native_outcome() and client.solved() is None
    assert "native_outcome" not in worker._outcome()
    assert ("view_env" in {tool.name for tool in client.list_tools()}) == bool(
        client.tool_capabilities["rgb_cameras"]
    )
    move = {"action": "move_to", "xyz": [10, 0, 30]}
    assert not client.execute_tool("execute_actions", {"actions": [move]}).is_error
    assert client.native_outcome() == kit.native_outcome()
    assert client.native_outcome()["episode_return"] == 4.0


@pytest.mark.parametrize("connected", ["state"], indirect=True)
def test_remote_preserves_native_steps_after_last_observation(connected):
    client, kit, _ = connected
    kit._primitives.step_controllers()
    client.close()
    assert client.state.latest_record().extras["episode_return"] == 0
    assert client.native_outcome()["episode_return"] == 2
    assert client.native_outcome()["success"] is None


# Native tests import simulator dependencies only when explicitly enabled.
NATIVE_ENABLED = os.environ.get("RPENT_RUN_MANISKILL_INTEGRATION") == "1"
if NATIVE_ENABLED:
    from rlinf.agents.rpent.maniskill_tasks import CARTESIAN_TASKS
else:
    CARTESIAN_TASKS = ()


@pytest.fixture
def open_episode(tmp_path):
    if not NATIVE_ENABLED:
        pytest.skip("set RPENT_RUN_MANISKILL_INTEGRATION=1 for native simulation")
    init_output_dir(tmp_path)
    with ExitStack() as release:

        def start(env_id):
            seed = 7 if env_id == "PickCube-v1" else 0
            task = {"environment": "maniskill", "task_name": env_id, "seed": seed}
            request = tmp_path / "request.json"
            request.write_text(json.dumps({"task": task}))
            args = rl_episode.parse_args(
                ["--output-dir", str(tmp_path), "--request", str(request)]
            )
            return rl_episode.start_environment(args, task, release)[0]

        yield start


@pytest.mark.parametrize(
    "env_id", ["MS-CartpoleBalance-v1", "AnymalC-Reach-v1", "RotateValveLevel0-v1"]
)
def test_excluded_tasks_rejected_before_simulation(env_id, open_episode):
    with pytest.raises(ValueError, match="Unsupported ManiSkill pose task"):
        open_episode(env_id)


@pytest.mark.parametrize("env_id", sorted(CARTESIAN_TASKS))
def test_native_episode_caps_coordinates_and_horizon(env_id, open_episode, monkeypatch):
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.registration import REGISTERED_ENVS
    from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv

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
        assert state["goal"]["xyz"] == pytest.approx(
            100 * sim.coordinates.points(goal), abs=0.01
        )
        assert state["is_grasped"] is False and state["cube_size_cm"] == pytest.approx(
            4
        )
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


@pytest.mark.parametrize(
    "env_id",
    ["UnitreeG1PlaceAppleInBowl-v1", "UnitreeG1TransportBox-v1", "InsertFlower-v1"],
)
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
    assert [item["status"] for item in result["results"]] == ["reached", "reached"], (
        result
    )
    final = result["final_state"]
    measured = final[side] if sim.dual_arm else final
    assert measured["opening"] == pytest.approx(0.5, abs=0.08)
    if sim.dual_arm:
        assert final["right"]["pose"]["xyz"] == pytest.approx(
            state["right"]["pose"]["xyz"], abs=0.3
        )

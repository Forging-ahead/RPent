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

"""ManiSkill worker schemas, seeded cameras, and native action termination."""

import asyncio
import io
import json
import os
from functools import partial
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from PIL import Image

from robots.rl.perception import RecordedPerception
from robots.rl.rollout import run_episode
from robots.rl.server import (
    SimPoolManager,
    _WorkerSessionHost,
)
from robots.rl.toolkit import episode_tools


@pytest.mark.parametrize(
    ("profile", "allowed", "rejected"),
    [
        (
            "arm",
            {"action": "set_gripper", "opening": 0},
            {"action": "move_lift", "delta_cm": 2},
        ),
        (
            "pen",
            {"action": "move_to", "xyz": [2, 4, 8]},
            {"action": "set_gripper", "opening": 0},
        ),
        (
            "dual",
            {"action": "set_gripper", "arm": "left", "opening": 0},
            {"action": "set_gripper", "opening": 0},
        ),
        (
            "mobile",
            {"action": "move_base", "vx": 5, "vy": 0, "yaw_rate": 0, "duration_s": 0.5},
            {"action": "move_base", "vx": 0, "vy": 5, "yaw_rate": 0, "duration_s": 0.5},
        ),
    ],
)
def test_schema_follows_active_robot(profile, allowed, rejected):
    perception = RecordedPerception()
    perception.tool_capabilities = {
        "rgb_cameras": ("base", "wrist"),
        "depth_cameras": ("base",),
        "segment": False,
        "gripper": profile != "pen",
        "dual_arm": profile == "dual",
        "mobile": profile == "mobile",
    }
    tools = {t.name: t for t in episode_tools(None, "maniskill", perception=perception)}
    assert set(tools) == {"view_env", "back_project", "execute_actions"}
    assert tools["view_env"].input_schema["properties"]["camera"]["enum"] == [
        "base",
        "wrist",
        "all",
    ]
    assert tools["back_project"].input_schema["properties"]["camera"]["enum"] == [
        "base",
        None,
    ]
    batch = tools["execute_actions"]
    batch.validate_arguments({"actions": [allowed]})
    with pytest.raises(ValueError):
        batch.validate_arguments({"actions": [rejected]})


def test_native_failure_reaches_rollout(monkeypatch):
    declaration = {"name": "view_env", "parameters": {"type": "object"}}
    call = {"id": "c", "function": {"name": "view_env", "arguments": "{}"}}
    statuses = {
        "/sessions/open": {
            "session_id": "s",
            "environment": "maniskill",
            "init_state_id": 0,
            "system_prompt": "",
            "user_message": "",
            "initial_observation": "{}",
            "tools": [{"type": "function", "function": declaration}],
            "done": False,
            "success": False,
            "truncated": False,
        },
        "/chat/completions": {"choices": [{"message": {"tool_calls": [call]}}]},
        "/sessions/s/execute": {
            "text": "{}",
            "done": True,
            "success": False,
            "truncated": False,
            "terminated": True,
            "fail": True,
            "episode_return": -1.5,
        },
        "/sessions/s/close": {"closed": True},
    }
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json=statuses[req.url.path])
    )
    monkeypatch.setattr(
        httpx, "AsyncClient", partial(httpx.AsyncClient, transport=transport)
    )
    result = asyncio.run(
        run_episode(
            model_url="http://model",
            model_name="test",
            sim_url="http://sim",
            task={"seed": 0},
        )
    )
    assert result["termination_reason"] == "environment_failure"
    assert result["terminated"] and result["fail"] and not result["truncated"]
    assert not result["success"] and result["episode_return"] == -1.5


def test_recycling_closes_maniskill_recorder():
    async def scenario():
        manager = SimPoolManager(
            environment="maniskill",
            gpus=[0],
            num_workers=1,
            sim_python="python",
            sam3_endpoint=None,
            max_episodes_per_worker=1,
        )
        worker = Mock(
            episodes=0, replaced=False, alive=True, busy=False, rpc=AsyncMock()
        )
        worker.proc.poll.return_value = None
        manager._sessions["s"] = worker
        manager._replace_worker = AsyncMock()
        try:
            assert (await manager.close_session("s"))["recycled"]
            worker.rpc.assert_awaited_once_with({"op": "close"})
            await asyncio.sleep(0)
            manager._replace_worker.assert_awaited_once_with(worker)
        finally:
            await manager.shutdown()

    asyncio.run(scenario())


@pytest.mark.skipif(
    os.environ.get("RPENT_TEST_MANISKILL") != "1",
    reason="requires a ManiSkill GPU runtime",
)
def test_native_worker_seed_camera_and_timeout(tmp_path):
    host = _WorkerSessionHost("maniskill", None, 120, None, None, 0)
    req = {
        "task": {"task_name": "PickCube-v1", "seed": 0},
        "max_episode_steps": 1,
        "image_size": 256,
        "record_artifacts": True,
        "output_dir": str(tmp_path),
    }
    try:
        opened = host.open(req)
        assert opened["init_state_id"] == 0 and not opened["done"]
        measured = json.loads(opened["initial_observation"])["state"]["pose"]["xyz"]
        rgb = host.execute({"name": "view_env", "arguments": {"camera": "base_camera"}})
        assert Image.open(io.BytesIO(rgb["images"][0])).size == (256, 256)
        target = [*measured[:2], measured[2] + 4]
        move = {"action": "move_to", "xyz": target}
        result = host.execute(
            {"name": "execute_actions", "arguments": {"actions": [move, move]}}
        )
        assert result["done"] and result["truncated"] and not result["success"]
        assert not result["terminated"]
        assert isinstance(result["episode_return"], float)
        assert (
            host._toolkit._state.get().extras["episode_return"]
            == result["episode_return"]
        )
        receipt = json.loads(result["text"])
        assert receipt["executed_count"] == 1
        assert receipt["results"][0]["env_steps"] == 1
        assert len(host._toolkit._primitives.frames) == 2
        reopened = host.open({**req, "record_artifacts": False})
        assert (
            json.loads(reopened["initial_observation"])["state"]["pose"]["xyz"]
            == measured
        )
    finally:
        host.close()

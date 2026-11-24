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


@pytest.mark.parametrize("env_id", sorted(REGISTERED_ENVS))
def test_native_environment_through_episode_entrypoint(env_id, tmp_path):
    """Exercise the same tools and task routing used by RLinf's episode worker."""
    task = {"environment": "maniskill", "task_name": env_id, "seed": 0}
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"task": task}))
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--request",
            str(request),
            "--max-episode-steps",
            "1",
        ]
    )
    init_output_dir(tmp_path)
    with contextlib.ExitStack() as release:
        try:
            toolkit, _ = start_environment(args, task, release)
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
        pose = state["left"]["pose"] if caps["dual_arm"] else state["pose"]
        x, y, z = pose["xyz"]
        move = {"action": "move_to", "xyz": [x, y, z + 2]}
        if caps["dual_arm"]:
            move["arm"] = "left"
        actions = [move, move]
        outcome = toolkit.execute_tool("execute_actions", {"actions": actions})
        assert outcome.data.get("executed_count") == 1, outcome.data
        metrics = toolkit.native_outcome()
        assert metrics["terminated"] or metrics["truncated"]
        assert metrics["success"] is None or type(metrics["success"]) is bool
        assert math.isfinite(metrics["episode_return"])
        assert toolkit.state.latest_record().extras["env_steps"] == 1

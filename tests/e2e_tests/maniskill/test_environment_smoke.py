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

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: E402, F401 - populates the native registry
import numpy as np  # noqa: E402
from mani_skill.utils.registration import REGISTERED_ENVS  # noqa: E402

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
        toolkit, _ = start_environment(args, task, release)
        opening = toolkit.initial_observation().data
        caps = opening["sensors"]
        tools = {definition.name for definition in toolkit.list_tools()}
        assert ("view_env" in tools) == bool(caps["rgb_cameras"])
        assert ("back_project" in tools) == bool(caps["depth_cameras"])
        reference_cameras = {
            "UnitreeG1Stand-v1": [],
            "MS-AntWalk-v1": ["side_cam"],
            "MS-HopperHop-v1": ["cam0"],
        }
        if env_id in reference_cameras:
            assert caps["rgb_cameras"] == reference_cameras[env_id]
        for camera in caps["rgb_cameras"]:
            response = toolkit.execute_tool("view_env", {"camera": camera})
            assert len(response.images) == 1, response.data

        # Sample from the controller bounds advertised to the model, as in Gym's demo.
        space = opening["action_space"]
        box = gym.spaces.Box(
            low=np.array(
                [v if v is not None else -np.inf for v in space["low"]],
                dtype=np.float32,
            ),
            high=np.array(
                [v if v is not None else np.inf for v in space["high"]],
                dtype=np.float32,
            ),
            seed=0,
        )
        assert list(box.shape) == caps["action_shape"]
        actions = [box.sample().tolist() for _ in range(2)]
        outcome = toolkit.execute_tool("execute_actions", {"actions": actions})
        assert outcome.data.get("executed_count") == 1, outcome.data
        metrics = toolkit.native_outcome()
        assert metrics["terminated"] or metrics["truncated"]
        assert metrics["success"] is None or type(metrics["success"]) is bool
        if env_id == "MS-AntWalk-v1":
            assert metrics["success"] is None
        assert math.isfinite(metrics["episode_return"])
        assert toolkit.state.latest_record().extras["env_steps"] == 1

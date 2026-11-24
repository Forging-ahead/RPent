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


def test_pick_cube_uses_rlinf_env():
    from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv

    from robots.rl.maniskill import _make_env

    env, task = _make_env("PickCube-v1", None, 1, seed=7)
    with contextlib.closing(env):
        assert task == "PickCube-v1"
        assert isinstance(env, ManiskillEnv)
        assert env.seed == 7
        obs, _ = env.reset()
        assert obs["sensor_data"]["base_camera"]["rgb"].ndim == 3
        _, reward, _, truncated, info = env.step(env.action_space.sample())
        assert bool(truncated)
        assert int(info["episode"]["episode_len"]) == 1
        assert float(info["episode"]["return"]) == pytest.approx(float(reward))


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

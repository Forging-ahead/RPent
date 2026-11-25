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

"""One episode conversation with explicit perception and measured action feedback."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from robots.rl.contracts import CAMERAS, EPISODE_CONTRACT_VERSION
from robots.rl.images import NATIVE_SIZE
from robots.rl.perception import RecordedPerception
from robots.rl.units import CENTIMETERS
from rpent.dashboard.events import DashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import ToolResult
from rpent.tools.toolkit import Toolkit
from rpent.utils.serialization import write_json


def primitive_owner(environment: str):
    """Select the same primitive owner class for local and remote execution."""
    if environment == "maniskill":
        from robots.rl.maniskill import ManiSkillPrimitives

        return ManiSkillPrimitives
    if environment == "libero":
        from robots.rl.tools import LiberoRLPrimitives

        return LiberoRLPrimitives
    if environment == "robodojo":
        from robots.rl.robodojo import RoboDojoPrimitives

        return RoboDojoPrimitives
    from robots.rl.backends import (
        RoboCasaGeneralPrimitives,
        RoboTwinGeneralPrimitives,
    )

    if environment == "robocasa":
        return RoboCasaGeneralPrimitives
    if environment == "robotwin":
        return RoboTwinGeneralPrimitives
    raise ValueError(f"Unsupported episode environment: {environment}")


def episode_tools(
    toolkit, environment, max_actions, perception=None, *, capabilities=None
):
    """Build the same model schema without importing native adapters on the client."""
    from robots.rl.actions import (
        ActionBatch,
        DualArmBatch,
        JointBatch,
        MobileBatch,
        ToolTipBatch,
    )
    from robots.rl.contracts import coordinate_definition, strict_tool
    from robots.rl.units import model_tool

    owner = RecordedPerception if perception is None else perception
    if environment == "maniskill":
        rgb, depth = capabilities["rgb_cameras"], capabilities["depth_cameras"]
        views = []
        if rgb:
            views.append((owner.view_env, rgb))
            if capabilities["segment"]:
                views.append((owner.segment, rgb))
        if depth:
            views.append((owner.back_project, depth))
        batch_type = DualArmBatch if capabilities["dual_arm"] else ActionBatch
        if capabilities.get("joint_control"):
            batch_type = JointBatch
        elif capabilities.get("mobile"):
            batch_type = MobileBatch
        elif not capabilities["gripper"]:
            batch_type = ToolTipBatch
    else:
        cameras = CAMERAS[environment]
        views = [
            (tool, cameras)
            for tool in (owner.view_env, owner.segment, owner.back_project)
        ]
        batch_type = {
            "libero": ActionBatch,
            "robocasa": MobileBatch,
            "robotwin": DualArmBatch,
            "robodojo": DualArmBatch,
        }[environment]
    batch = batch_type(toolkit).execute_actions
    schema = batch.input_schema
    if environment == "maniskill" and capabilities.get("mobile"):
        lateral = schema["$defs"]["BaseMove"]["properties"]["vy"]
        lateral.update(
            minimum=0,
            maximum=0,
            description="Fetch differential drive requires zero lateral velocity. Turn before driving sideways.",
        )
    if environment == "maniskill" and capabilities.get("joint_control"):
        count = len(capabilities["joint_control"]["action_order"])
        schema["$defs"]["JointDrive"]["properties"]["values"].update(
            minItems=count, maxItems=count
        )
    schema["properties"]["actions"]["maxItems"] = max_actions
    batch = replace(
        batch,
        _input_schema=schema,
        description=batch.description.replace("1..K", f"1..{max_actions}"),
    )
    for declaration, cameras in [*views, (batch, ())]:
        yield model_tool(
            strict_tool(
                coordinate_definition(declaration),
                cameras=cameras,
            ),
        )


class RLToolkit(Toolkit):
    """Expose explicit perception and ordered actions over one simulator scene."""

    planner_builtin_tools = False
    api_environment_context = False

    def __init__(
        self,
        *,
        runtime_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
        memory: MemoryManager,
        output_dir: Path | str,
        environment: str = "libero",
        image_size: int = 1024,
        max_actions_per_call: int = 3,
        reasoning_history: int = -1,
        image_history: int = -1,
        reasoning: bool = True,
    ) -> None:
        self.reasoning = reasoning
        self.max_actions_per_call = max_actions_per_call
        self.reasoning_history, self.image_history = reasoning_history, image_history
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.workflow_stats = {"perception_seconds": 0.0}
        self._started = time.perf_counter()
        self._native_success_seconds = None
        state = EnvState(output_dir)
        super().__init__(dashboard_events=dashboard_events, state=state, memory=memory)
        self._images_sent = 0
        self._images_lock = threading.Lock()
        self.environment = environment
        self._primitives = primitive_owner(environment)(
            state=state, check_cancelled=self.raise_if_cancelled, **runtime_kwargs
        )
        native = self._primitives
        self.tool_capabilities = None
        try:
            if environment == "maniskill":
                native.display_size = image_size
                self._task = native.task
                self.tool_capabilities = {
                    "rgb_cameras": list(native.cameras),
                    "depth_cameras": list(native.depth_cameras),
                    "segment": bool(native.cameras) and native._sam3 is not None,
                    "dual_arm": native.dual_arm,
                    "mobile": bool(getattr(native, "mobile", False)),
                    "gripper": native.has_gripper,
                }
                if getattr(native, "joint_drive", None) is not None:
                    self.tool_capabilities["joint_control"] = native.joint_drive.layout
            else:
                if environment in {"libero", "robocasa"} and image_size != NATIVE_SIZE:
                    native.display_size = image_size
                self._task = native.env.get_task_language()
            self.action_error = None
            self._publish_step(native.record_observation())
            self._recorded_env_steps = 0
            self.add_tools(
                episode_tools(
                    self,
                    environment,
                    max_actions_per_call,
                    perception=native,
                    capabilities=self.tool_capabilities,
                )
            )
        except BaseException:
            if environment == "maniskill":
                native.env.close()
            raise

    def frame_version(self, *, refresh: bool = False) -> int:
        """Read the measured frame under the episode's exclusive action lock."""
        if refresh and self.environment == "robocasa":
            self._primitives._refresh_coordinates()
        return getattr(getattr(self._primitives, "coordinates", None), "version", 0)

    def _model_feedback(self, result: ToolResult) -> ToolResult:
        result.data["coordinates"] = self._primitives.coordinates.description()
        result.data = CENTIMETERS.payload(result.data)
        result.data["position_unit"] = "cm"
        return result

    def _register_common_tools(self) -> None:
        """Register no file tools; rollouts observe only through robot tools."""

    def _robot_state(self) -> dict[str, Any]:
        if self.environment == "maniskill":
            return {
                **self._primitives.robot_state(),
                "terminated": self._primitives.terminated,
                "truncated": self._primitives.truncated,
            }
        if self.environment != "libero":
            return {
                **self._primitives.robot_state(),
                "terminated": self._primitives.native_success(),
                "truncated": self._primitives.env_steps
                >= self._primitives._max_episode_steps,
            }
        env = self._primitives.env
        return {
            **self._primitives.measure().state(),
            "terminated": env.terminated,
            "truncated": env.truncated,
        }

    def initial_observation(self) -> ToolResult:
        """Return the task and measured initial state for the opening message."""
        gate = os.environ.get("RPENT_START_GATE")
        if gate:
            write_json(
                self.output_dir / "READY.json",
                {
                    "pid": os.getpid(),
                    "time": time.time(),
                    "gate": gate,
                    "task": self._task,
                },
            )
            deadline = time.monotonic() + float(
                os.environ.get("RPENT_START_GATE_TIMEOUT_S", "1200")
            )
            while not Path(gate).exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("batch start barrier timed out")
                time.sleep(0.1)
        self._started = time.perf_counter()
        self._started_at_wall = time.time()
        self._state.save(
            "observation-start.json",
            {"started_at": self._started_at_wall, "gate": gate},
            step=None,
        )
        step = self._state.latest_step
        observation = ToolResult(
            data={
                "task": self._task,
                "state": {"step": step, **self._robot_state()},
            },
        )
        if self.environment == "maniskill":
            observation.data["sensors"] = self.tool_capabilities
        observation = self._model_feedback(observation)
        self._count_images(observation)
        self._state.save(
            "planner_audit.json",
            {
                "episode_contract_version": EPISODE_CONTRACT_VERSION,
                "tools": [
                    {
                        "name": definition.name,
                        "description": definition.description,
                        "input_schema": definition.input_schema,
                    }
                    for definition in self.list_tools()
                ],
                "initial_observation": observation.data,
                "initial_images": len(observation.images),
            },
            step=None,
        )
        return observation

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Run one tool and count the images returned to the planner."""
        tool_started = time.perf_counter()
        declaration = self._tools.get(name)
        if declaration is None:
            return super().execute_tool(name, input_dict)
        try:
            parameters = declaration.validate_arguments(input_dict)
            arguments = {
                field: getattr(parameters, field)
                for field in type(parameters).model_fields
            }
            from robots.rl.contracts import validate_relations

            validate_relations(name, arguments)
        except (ValueError, TypeError) as exc:
            from robots.rl.contracts import argument_hint

            hint = argument_hint(name, self.environment, declaration)
            return ToolResult(
                data={"error": str(exc)},
                model_text=f"Invalid {name} arguments: {exc}. {hint} No tool executed; correct the arguments and try again.",
            )
        result = self._execute_validated(declaration, input_dict, arguments)
        if self.solved() and self._native_success_seconds is None:
            self._native_success_seconds = time.perf_counter() - self._started
        elapsed = time.perf_counter() - tool_started
        timings = self.workflow_stats.setdefault("tool_seconds_by_name", {})
        timings[name] = timings.get(name, 0.0) + elapsed
        if name in {"segment", "back_project"}:
            self.workflow_stats["perception_seconds"] += elapsed
        result = self._model_feedback(result)
        self._count_images(result)
        return result

    def _count_images(self, result: ToolResult) -> None:
        with self._images_lock:
            self._images_sent += len(result.images)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
        start_frame_version: int | None = None,
    ) -> ToolResult:
        """Return the measured outcome of a robot action.

        A new observation is recorded only when the action stepped the
        environment; otherwise the latest step number is reused.
        """
        if command["action"] == "execute_actions":
            return ToolResult(data=result)
        if self.infrastructure_error is not None:
            # The native action may have executed before its acknowledgement
            # was lost. Preserve the last observation; do not label a new
            # capture or a partial receipt as an authoritative post-action state.
            return ToolResult(
                data={
                    **result,
                    "status": "error",
                    "error": "native action outcome is uncertain",
                    "state_stale": True,
                    "start_frame_version": start_frame_version,
                    "step": self._state.latest_step,
                }
            )
        advanced = self._primitives.env_steps > self._recorded_env_steps
        step = self._state.latest_step + 1 if advanced else self._state.latest_step
        if "error" in result:
            data: dict[str, Any] = {"status": "error", "error": result["error"]}
        else:
            data = {"status": result["status"]}
            if "reason" in result:
                data["reason"] = result["reason"]
        if self.environment != "libero":
            # Native adapters also report per-arm errors and physical limits.
            data.update(result)
        for key in (
            "position_error_m",
            "rotation_error_rad",
            "opening_error",
            "env_steps",
        ):
            if key in result:
                data[key] = result[key]
        data["step"] = step
        if advanced or data["status"] != "error":
            data.update(self._robot_state())
        data["start_frame_version"] = start_frame_version
        data["frame_version"] = self.frame_version()
        if advanced:
            # Log the payload the planner receives, not the primitive's internals.
            capture_started = time.perf_counter()
            self._primitives.record_observation(
                command=command, result=data, elapsed_s=elapsed_s
            )
            self.workflow_stats["observation_capture_seconds"] = (
                self.workflow_stats.get("observation_capture_seconds", 0.0)
                + time.perf_counter()
                - capture_started
            )
            self._recorded_env_steps = self._primitives.env_steps
        return ToolResult(data=data)

    def solved(self) -> bool | None:
        """Return the environment's latched success flag."""
        if self.environment != "libero":
            return self._primitives.native_success()
        return self._primitives.env.terminated

    def native_outcome(self) -> dict[str, Any]:
        """Return the current native ManiSkill episode metrics, including partial runs."""
        return (
            self._primitives.native_outcome() if self.environment == "maniskill" else {}
        )

    @property
    def infrastructure_error(self) -> dict | None:
        native_error = getattr(self._primitives, "infrastructure_error", None)
        return native_error if native_error is not None else self.action_error

    def completion_result(self) -> dict | None:
        """End the conversation before another request after a native terminal."""
        if self.infrastructure_error is not None:
            return {"status": "error", "reason": "infrastructure_error"}
        if self._primitives.ended:
            if self.environment == "maniskill":
                return {"status": "completed", **self.native_outcome()}
            success = self.solved()
            return {"status": "completed", "success": success, "truncated": not success}
        return None

    def close(self) -> None:
        """Save the episode video and the rollout's observation counts."""
        super().close()
        try:
            self._save_rollout()
        finally:
            if self.environment == "maniskill":
                self._primitives.env.close()

    def _save_rollout(self) -> None:
        if self._primitives.frames:
            self._state.save("episode.mp4", self._primitives.frames, step=None, fps=20)
        contract = {
            "position_unit": "cm",
            "coordinate_contract": "episode_coordinates_v2",
            **self.native_outcome(),
        }
        self._state.save(
            "rollout.json",
            {
                **contract,
                "episode_contract_version": EPISODE_CONTRACT_VERSION,
                "reasoning": self.reasoning,
                "image_size": self.image_size,
                "max_actions_per_call": self.max_actions_per_call,
                "reasoning_history": self.reasoning_history,
                "image_history": self.image_history,
                "native_success_seconds": self._native_success_seconds,
                "started_at_wall": getattr(self, "_started_at_wall", None),
                **self.workflow_stats,
                "images_sent": self._images_sent,
                "images_generated_for_feedback": self._images_sent,
                "env_steps": self._primitives.env_steps,
                "observations": self._state.latest_step + 1,
            },
            step=None,
        )

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

"""RL conversation boundaries and response gates on the native API agent loop."""

from __future__ import annotations

import time
from dataclasses import replace

from pydantic_ai import ModelRetry, ToolCallPart
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, SystemPromptPart


class WorkflowCapability(AbstractCapability):
    """Validate complete replies before executing any tool."""

    def __init__(self, toolkit, system_prompt=None) -> None:
        self.toolkit = toolkit
        self.original_prompt = system_prompt
        self.joint_prompt = None
        if (getattr(toolkit, "tool_capabilities", None) or {}).get("joint_control"):
            from robots.rl.prompt_bundle import joint_prompt

            self.joint_prompt = joint_prompt(toolkit)
        self.allowed = {item.name: item for item in toolkit.list_tools()}
        self.stats = toolkit.workflow_stats
        self.stats.update(invalid_tool_responses=0, model_seconds=0.0)

    async def before_model_request(self, ctx, request_context):
        """Allow a final reply or a single validated tool call."""
        if self.joint_prompt is not None:
            # The CLI and Dashboard render the generic prompt before the simulator starts.
            # Replace that specific prompt in the request view after capability negotiation.
            for index, message in enumerate(request_context.messages):
                if isinstance(message, ModelRequest) and any(
                    isinstance(part, SystemPromptPart)
                    and part.content == self.original_prompt
                    for part in message.parts
                ):
                    request_context.messages[index] = replace(
                        message,
                        parts=[
                            replace(part, content=self.joint_prompt)
                            if isinstance(part, SystemPromptPart)
                            and part.content == self.original_prompt
                            else part
                            for part in message.parts
                        ],
                    )
                    break
        request_context.model_settings = {
            **(request_context.model_settings or {}),
            "parallel_tool_calls": False,
        }
        from rpent.planner.qwen import is_qwen_model

        prepared = request_context.model_request_parameters
        definitions = [*prepared.function_tools, *prepared.output_tools]
        joint_only = (
            self.joint_prompt is not None
            and len(definitions) == 1
            and definitions[0].name == "execute_actions"
        )
        bounded = is_qwen_model(getattr(request_context.model, "model_name", "")) and (
            joint_only
            or any(
                t.parameters_json_schema.get("type") == "object"
                and not t.parameters_json_schema.get("required")
                and t.parameters_json_schema.get("additionalProperties") is False
                for t in definitions
            )
        )
        if bounded:
            from robots.rl.qwen_grammar import bounded_tool_format

            settings = request_context.model_settings
            settings["extra_body"] = {
                **(settings.get("extra_body") or {}),
                "tool_choice": "auto",
                "response_format": bounded_tool_format(
                    definitions,
                    parallel=False,
                    require_tool=False,
                    fractional_bounds_at_execution=True,
                ),
            }
        self.request_started = time.perf_counter()
        return request_context

    async def after_model_request(self, ctx, *, request_context, response):
        """Reject the whole reply before any sibling tool can change state."""
        elapsed = time.perf_counter() - self.request_started
        self.stats["model_seconds"] += elapsed
        trace = getattr(request_context.model, "request_trace", None)
        if trace is not None:
            trace.response(response)
        calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
        error = None
        if response.state != "complete" or response.finish_reason == "length":
            raise UnexpectedModelBehavior(
                "Token limit reached before any response was generated"
            )
        if len(calls) > 1:
            error = "Use one tool call per response."
        if error is None:
            for call in calls:
                try:
                    if call.tool_name not in self.allowed:
                        raise ValueError(f"unknown tool {call.tool_name}")
                    arguments = call.args_as_dict()
                    self.allowed[call.tool_name].validate_arguments(arguments)
                    from robots.rl.contracts import validate_relations

                    validate_relations(call.tool_name, arguments)
                except (ValueError, TypeError) as exc:
                    from robots.rl.contracts import argument_hint

                    hint = argument_hint(
                        call.tool_name,
                        getattr(self.toolkit, "environment", "libero"),
                        self.allowed.get(call.tool_name),
                    )
                    error = f"Invalid tool arguments: {exc}. {hint}"
                    break
        if error:
            self.stats["invalid_tool_responses"] += 1
            feedback = error + " No tool executed."
            raise ModelRetry(feedback)
        return response

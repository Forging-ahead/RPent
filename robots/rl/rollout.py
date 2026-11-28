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

"""Standalone async rollout, evaluation, and resumable SFT trajectory collector.

Uses the exact same Harness v2 ``EpisodeConversation`` (step-0 opening images,
compact feedback, optional ``jaw_mm``, ``execute_actions(..., observe=...)``,
and ``base``/``dynamic`` compaction) and tool-validation loop as RLinf training
so standalone evaluation and SFT distillation stay 100% aligned with RL.

Key SFT collection capabilities:
- Per-episode atomic persistence (``episodes/<episode_key>/trajectory.json`` +
  deduplicated ``images/<fid>.png`` + append-only ``manifest.jsonl``) so runs
  can be interrupted and resumed via ``--resume`` with zero lost work.
- Sticky least-loaded multi-endpoint routing across SGLang continuous-batching
  replicas (``--model-url url1,url2,...`` or ``--model-urls-file``) to maximize
  RadixCache hits and GPU batch saturation.
- Provider-aware OpenAI-compatible client supporting both local SGLang models
  (e.g. ``Qwen3.8-27B``) and remote APIs (e.g. ``deepseek-v4.1-flash``) with
  exponential backoff on HTTP 429/5xx/timeouts.
- Intra-task sharding (``--shard i/N``) and round-robin task interleaving
  (``--interleave-tasks``) so multiple teachers cover 100% of tasks without
  seed overlap.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import OrderedDict
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Literal

import httpx

from robots.rl.conversation import CompactionConfig, EpisodeConversation
from robots.rl.server import BINARY_EXECUTE_MEDIA_TYPE, unpack_execute_response
from rpent.utils.logging import get_logger

logger = get_logger("rl_rollout")

RETRYABLE_LLM_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
RETRYABLE_SIM_STATUS = frozenset({408, 429, 500, 502, 503, 504})
ProviderMode = Literal["auto", "sglang", "openai"]


def parse_model_urls(
    model_url: str | None = None, model_urls_file: Path | None = None
) -> list[str]:
    """Parse one or more OpenAI-compatible ``/v1`` base URLs."""
    urls: list[str] = []
    if model_url:
        for item in model_url.split(","):
            cleaned = item.strip().rstrip("/")
            if cleaned:
                urls.append(cleaned)
    if model_urls_file is not None:
        for line in model_urls_file.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip().rstrip("/")
            if cleaned and not cleaned.startswith("#"):
                urls.append(cleaned)
    deduped = list(dict.fromkeys(urls))
    if not deduped:
        raise ValueError("Provide at least one URL via --model-url or --model-urls-file")
    return deduped


def detect_provider(
    provider: ProviderMode,
    *,
    model_name: str,
    model_urls: Sequence[str],
    api_key: str | None,
) -> Literal["sglang", "openai"]:
    """Select wire protocol flavor (``sglang`` vs pure ``openai``)."""
    if provider in {"sglang", "openai"}:
        return provider
    lower_model = model_name.lower()
    lower_urls = " ".join(u.lower() for u in model_urls)
    if api_key or "deepseek" in lower_model or any(
        hint in lower_urls
        for hint in ("infini-ai", "deepseek", "openai.com", "openrouter", "maas")
    ):
        return "openai"
    return "sglang"


def split_inline_thinking(
    content: str | None, reasoning_content: str | None
) -> tuple[str | None, str | None]:
    """Extract inline ``<think>...</think>`` when a server omits ``reasoning_content``."""
    if reasoning_content is not None or not isinstance(content, str):
        return content, reasoning_content
    if "</think>" not in content:
        return content, None
    before, _, after = content.partition("</think>")
    if "<think>" in before:
        reasoning = before.rsplit("<think>", 1)[-1].strip()
    else:
        reasoning = before.strip()
    rest = after.lstrip("\n")
    return (rest or None), reasoning


def sanitize_episode_key(raw: str) -> str:
    """Convert an arbitrary query identifier into a safe directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw.strip()).strip("._-")
    return cleaned or "episode"


def _task_group_key(task: dict[str, Any], query_id: str) -> str:
    for field in ("task_name", "task_id", "bddl_file_name", "task"):
        val = task.get(field)
        if val is not None and str(val).strip():
            return str(val)
    return query_id


def parse_shard_spec(spec: str | None) -> tuple[int, int]:
    """Parse ``i/N`` (0-based shard index and total shard count)."""
    if not spec:
        return 0, 1
    parts = spec.strip().split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid --shard value {spec!r}; expected '<index>/<total>'")
    shard_idx, num_shards = int(parts[0]), int(parts[1])
    if num_shards <= 0 or not (0 <= shard_idx < num_shards):
        raise ValueError(
            f"Invalid --shard {spec!r}: require 0 <= index < total and total > 0"
        )
    return shard_idx, num_shards


def build_episode_schedule(
    rows: Sequence[dict[str, Any]],
    *,
    interleave_tasks: bool = True,
    shard_index: int = 0,
    num_shards: int = 1,
    target_count: int | None = None,
    limit: int | None = None,
    pass_offset: int = 0,
) -> list[dict[str, Any]]:
    """Build deterministic episode assignments with intra-task sharding and interleaving.

    Sharding is applied *inside* each task group before round-robin interleaving
    so every shard covers 100% of tasks with disjoint seeds/init states. When
    ``target_count`` exceeds the shard length, additional passes (``r1``, ``r2``,
    ...) cycle cleanly with distinct ``episode_key`` values.
    """
    parsed: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        task = (
            json.loads(row["answer"])
            if isinstance(row.get("answer"), str)
            else dict(row)
        )
        raw_qid = row.get("query_id")
        if not raw_qid:
            env_name = task.get("environment", "env")
            tname = _task_group_key(task, f"ep_{idx:05d}")
            seed = task.get("seed", task.get("init_state_id", idx))
            raw_qid = f"{env_name}-{tname}-s{seed}-i{idx}"
        parsed.append(
            {
                "source_idx": idx,
                "query_id": str(raw_qid),
                "task_group": _task_group_key(task, str(raw_qid)),
                "task": task,
            }
        )

    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for item in parsed:
        groups.setdefault(item["task_group"], []).append(item)

    sharded_groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for gname, items in groups.items():
        if num_shards > 1:
            subset = items[shard_index::num_shards]
            if not subset and items:
                subset = [items[shard_index % len(items)]]
        else:
            subset = list(items)
        if subset:
            sharded_groups[gname] = subset

    if interleave_tasks:
        base_order: list[dict[str, Any]] = []
        max_len = max((len(v) for v in sharded_groups.values()), default=0)
        for pos in range(max_len):
            for items in sharded_groups.values():
                if pos < len(items):
                    base_order.append(items[pos])
    else:
        base_order = [item for items in sharded_groups.values() for item in items]

    if not base_order:
        return []

    total = target_count if target_count is not None else len(base_order)
    if limit is not None:
        total = min(total, limit)

    schedule: list[dict[str, Any]] = []
    for seq_idx in range(total):
        pass_idx = pass_offset + (seq_idx // len(base_order))
        item = base_order[seq_idx % len(base_order)]
        ep_key = sanitize_episode_key(f"{item['query_id']}__r{pass_idx}")
        schedule.append(
            {
                "schedule_idx": seq_idx,
                "pass_idx": pass_idx,
                "episode_key": ep_key,
                "query_id": item["query_id"],
                "task_group": item["task_group"],
                "task": dict(item["task"]),
            }
        )
    return schedule


class EndpointPool:
    """Sticky least-loaded endpoint router maximizing SGLang RadixCache hits."""

    def __init__(self, urls: Sequence[str]) -> None:
        self.urls = [u.rstrip("/") for u in urls]
        self._inflight: dict[str, int] = {u: 0 for u in self.urls}

    def acquire(self) -> str:
        min_load = min(self._inflight.values())
        candidates = [u for u, load in self._inflight.items() if load == min_load]
        chosen = random.choice(candidates) if len(candidates) > 1 else candidates[0]
        self._inflight[chosen] += 1
        return chosen

    def release(self, url: str) -> None:
        if url in self._inflight:
            self._inflight[url] = max(0, self._inflight[url] - 1)

    def alternate(self, exclude_url: str) -> str:
        others = [u for u in self.urls if u != exclude_url]
        if not others:
            return exclude_url
        return min(others, key=lambda u: self._inflight[u])


def _retry_delay_s(attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return max(0.5, min(90.0, float(retry_after)))
        except ValueError:
            pass
    base = min(60.0, 2.0 * (2**attempt))
    return base + random.uniform(0.0, 1.5)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _save_episode_atomic(
    *,
    ep_dir: Path,
    episode_key: str,
    query_id: str,
    pass_idx: int,
    conv: EpisodeConversation,
    outcome: dict[str, Any],
    model_name: str,
    provider: str,
    model_url: str,
    rollout_config: dict[str, Any],
    usage_totals: dict[str, int],
) -> Path:
    """Persist deduplicated PNGs + atomic ``trajectory.json`` and ``outcome.json``."""
    ep_dir.mkdir(parents=True, exist_ok=True)
    img_dir = ep_dir / "images"
    images_by_id = conv.all_images_by_id()
    image_relpaths: dict[str, str] = {}
    if images_by_id:
        img_dir.mkdir(parents=True, exist_ok=True)
        for fid, png_bytes in images_by_id.items():
            img_path = img_dir / f"{fid}.png"
            if not img_path.exists():
                tmp_img = img_dir / f".{fid}.png.tmp.{os.getpid()}"
                tmp_img.write_bytes(png_bytes)
                os.replace(tmp_img, img_path)
            image_relpaths[fid] = f"images/{fid}.png"

    trajectory = {
        "schema_version": 2,
        "episode_key": episode_key,
        "query_id": query_id,
        "pass_idx": pass_idx,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "environment": conv.environment,
        "model": {
            "name": model_name,
            "provider": provider,
            "endpoint": model_url,
        },
        "rollout_config": rollout_config,
        "outcome": outcome,
        "usage": usage_totals,
        "tools": conv.tools,
        "images": image_relpaths,
        "compaction_events": conv.compaction_events,
        "full_messages": conv.full_wire_messages(inline_base64=False),
        "compacted_messages": conv.wire_messages(inline_base64=False),
    }
    traj_path = ep_dir / "trajectory.json"
    _atomic_write_json(traj_path, trajectory)
    _atomic_write_json(ep_dir / "outcome.json", outcome)
    return traj_path


def _load_completed_episodes(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Scan ``manifest.jsonl`` and ``episodes/*/trajectory.json`` for ``--resume``."""
    completed: dict[str, dict[str, Any]] = {}
    manifest = output_dir / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                key = rec.get("episode_key")
                if key:
                    traj_path = output_dir / "episodes" / key / "trajectory.json"
                    if traj_path.exists():
                        completed[str(key)] = rec
            except json.JSONDecodeError:
                continue

    episodes_root = output_dir / "episodes"
    if episodes_root.is_dir():
        for traj_path in episodes_root.glob("*/trajectory.json"):
            key = traj_path.parent.name
            if key in completed:
                continue
            try:
                data = json.loads(traj_path.read_text(encoding="utf-8"))
                out = data.get("outcome") or {}
                completed[key] = {
                    "episode_key": key,
                    "query_id": data.get("query_id"),
                    "environment": data.get("environment"),
                    "success": bool(out.get("success", False)),
                    "turns": int(out.get("turns", 0)),
                    "termination_reason": out.get("termination_reason"),
                    "error_kind": out.get("error_kind"),
                    "elapsed_s": out.get("elapsed_s", 0.0),
                    "trajectory_path": str(traj_path.relative_to(output_dir)),
                }
            except Exception:
                continue
    return completed


async def _post_llm_with_retry(
    client: httpx.AsyncClient,
    *,
    model_url: str,
    endpoint_pool: EndpointPool | None,
    payload: dict[str, Any],
    headers: dict[str, str],
    max_retries: int,
) -> tuple[dict[str, Any], str]:
    """Call ``/chat/completions`` with exponential backoff on 429/5xx/network errors."""
    active_url = model_url.rstrip("/")
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = await client.post(
                f"{active_url}/chat/completions", json=payload, headers=headers
            )
            if resp.status_code in RETRYABLE_LLM_STATUS and attempt < max_retries:
                delay = _retry_delay_s(attempt, resp.headers.get("Retry-After"))
                logger.warning(
                    "LLM HTTP %d from %s (attempt %d/%d); retrying in %.1fs",
                    resp.status_code,
                    active_url,
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            resp.raise_for_status()
            data = resp.json()
            if not data.get("choices"):
                raise ValueError(f"Empty choices in LLM response: {data}")
            return data, active_url
        except (httpx.TimeoutException, httpx.TransportError, ValueError) as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = _retry_delay_s(attempt)
            if endpoint_pool is not None and attempt >= 1 and len(endpoint_pool.urls) > 1:
                next_url = endpoint_pool.alternate(active_url)
                if next_url != active_url:
                    logger.warning(
                        "LLM transport error on %s (%s); failing over to %s",
                        active_url,
                        type(exc).__name__,
                        next_url,
                    )
                    active_url = next_url
            logger.warning(
                "LLM error from %s (%s: %s; attempt %d/%d); retrying in %.1fs",
                active_url,
                type(exc).__name__,
                exc,
                attempt + 1,
                max_retries + 1,
                delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def run_episode(
    *,
    model_url: str,
    model_name: str,
    sim_url: str,
    task: dict[str, Any],
    max_turns: int = 50,
    max_tokens: int = 16384,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    thinking: bool = True,
    compaction: CompactionConfig | None = None,
    image_size: int | None = None,
    max_actions_per_call: int = 3,
    include_jaw_mm: bool = True,
    record_artifacts: bool = False,
    output_dir: Path | None = None,
    request_timeout_s: float = 600.0,
    api_key: str | None = None,
    provider: ProviderMode = "auto",
    max_llm_retries: int = 8,
    max_infra_retries: int = 3,
    save_trajectory: bool = True,
    episode_key: str | None = None,
    query_id: str | None = None,
    pass_idx: int = 0,
    client: httpx.AsyncClient | None = None,
    endpoint_pool: EndpointPool | None = None,
) -> dict[str, Any]:
    """Run one standalone Harness v2 episode and optionally save ``trajectory.json``."""
    started = time.perf_counter()
    resolved_provider = detect_provider(
        provider,
        model_name=model_name,
        model_urls=[model_url],
        api_key=api_key,
    )
    llm_headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        llm_headers["Authorization"] = f"Bearer {api_key}"

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            timeout=request_timeout_s,
            trust_env=False,
            limits=httpx.Limits(keepalive_expiry=30.0),
        )

    try:
        base_seed = task.get("seed", task.get("init_state_id"))
        seeds = [base_seed, *(task.get("fallback_seeds") or ())]
        opened: dict[str, Any] | None = None
        initial_images: list[bytes] = []
        last_exc: Exception | None = None

        for infra_attempt in range(max_infra_retries + 1):
            for seed in seeds:
                attempt_task = dict(task)
                if seed is not None:
                    attempt_task["seed"] = seed
                    if "init_state_id" in attempt_task:
                        attempt_task["init_state_id"] = seed
                try:
                    open_resp = await client.post(
                        f"{sim_url.rstrip('/')}/sessions/open",
                        json={
                            "task": attempt_task,
                            "image_size": image_size,
                            "max_actions_per_call": max_actions_per_call,
                            "reasoning": thinking,
                            "include_jaw_mm": include_jaw_mm,
                            "record_artifacts": record_artifacts,
                            "output_dir": str(output_dir) if output_dir else None,
                        },
                        headers={"Accept": BINARY_EXECUTE_MEDIA_TYPE},
                    )
                    open_resp.raise_for_status()
                    if BINARY_EXECUTE_MEDIA_TYPE in open_resp.headers.get(
                        "content-type", ""
                    ):
                        opened, initial_images = unpack_execute_response(
                            open_resp.content
                        )
                    else:
                        opened = open_resp.json()
                        initial_images = [
                            base64.b64decode(b64)
                            for b64 in opened.get("images_b64") or ()
                        ]
                    break
                except (httpx.HTTPError, OSError) as exc:
                    last_exc = exc
            if opened is not None:
                break
            if infra_attempt < max_infra_retries:
                await asyncio.sleep(_retry_delay_s(infra_attempt))

        if opened is None:
            assert last_exc is not None
            raise last_exc
        sid = opened["session_id"]

        conv = EpisodeConversation(
            system_prompt=opened["system_prompt"],
            user_message=opened["user_message"],
            initial_observation_text=opened["initial_observation"],
            tools=opened["tools"],
            environment=opened["environment"],
            compaction=compaction,
            initial_images=initial_images,
            initial_image_cameras=opened.get("image_cameras") or (),
        )
        done = bool(opened["done"])
        success = bool(opened["success"])
        truncated = bool(opened["truncated"])
        reason = "success" if success else "planner_completed"
        error_kind: str | None = None
        error_msg: str | None = None
        turns = 0
        usage_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
        }
        active_model_url = model_url.rstrip("/")

        try:
            while not done and turns < max_turns:
                conv.maybe_compact()
                payload: dict[str, Any] = {
                    "model": model_name,
                    "messages": conv.wire_messages(inline_base64=True),
                    "tools": conv.tools,
                    "parallel_tool_calls": False,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                if resolved_provider == "sglang":
                    payload["top_k"] = top_k
                    payload["chat_template_kwargs"] = {
                        "enable_thinking": thinking,
                        "preserve_thinking": True,
                    }

                resp_json, active_model_url = await _post_llm_with_retry(
                    client,
                    model_url=active_model_url,
                    endpoint_pool=endpoint_pool,
                    payload=payload,
                    headers=llm_headers,
                    max_retries=max_llm_retries,
                )
                usage = resp_json.get("usage") or {}
                usage_totals["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                usage_totals["completion_tokens"] += int(
                    usage.get("completion_tokens") or 0
                )
                prompt_details = usage.get("prompt_tokens_details") or {}
                usage_totals["cached_tokens"] += int(
                    prompt_details.get("cached_tokens") or 0
                )

                choice = resp_json["choices"][0]
                turns += 1
                msg = choice["message"]
                finish = choice.get("finish_reason")
                tool_calls = msg.get("tool_calls") or []
                clean_content, clean_reasoning = split_inline_thinking(
                    msg.get("content"), msg.get("reasoning_content")
                )
                conv.append_assistant(
                    content=clean_content,
                    reasoning_content=clean_reasoning,
                    tool_calls=tool_calls or None,
                )
                if finish == "length":
                    error_kind, reason = "response_truncated", "response_truncated"
                    error_msg = "Token limit reached before completing response"
                    break
                if not tool_calls:
                    reason = "planner_completed"
                    break

                parsed_calls = [
                    {
                        "id": tc.get("id") or f"call_{turns}_{idx}",
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                    for idx, tc in enumerate(tool_calls)
                ]
                err_kind, err_msg, args, should_retry = conv.validate_or_retry(
                    parsed_calls
                )
                if err_kind is not None:
                    if should_retry:
                        continue
                    error_kind, error_msg, reason = err_kind, err_msg, err_kind
                    break

                exec_resp = None
                exec_exc: Exception | None = None
                for sim_attempt in range(max_infra_retries + 1):
                    try:
                        exec_resp = await client.post(
                            f"{sim_url.rstrip('/')}/sessions/{sid}/execute",
                            json={"name": parsed_calls[0]["name"], "arguments": args},
                            headers={"Accept": BINARY_EXECUTE_MEDIA_TYPE},
                        )
                        if (
                            exec_resp.status_code in RETRYABLE_SIM_STATUS
                            and sim_attempt < max_infra_retries
                        ):
                            await asyncio.sleep(_retry_delay_s(sim_attempt))
                            continue
                        exec_resp.raise_for_status()
                        break
                    except (httpx.HTTPError, OSError) as exc:
                        exec_exc = exc
                        exec_resp = None
                        if sim_attempt < max_infra_retries:
                            await asyncio.sleep(_retry_delay_s(sim_attempt))
                if exec_resp is None:
                    assert exec_exc is not None
                    raise exec_exc

                if BINARY_EXECUTE_MEDIA_TYPE in exec_resp.headers.get(
                    "content-type", ""
                ):
                    executed, images = unpack_execute_response(exec_resp.content)
                else:
                    executed = exec_resp.json()
                    images = []
                conv.append_tool_result(
                    tool_name=parsed_calls[0]["name"],
                    tool_call_id=parsed_calls[0]["id"],
                    text=executed["text"],
                    images=images,
                )
                success = bool(executed["success"])
                truncated = bool(executed["truncated"])
                done = bool(executed["done"])
                if executed.get("infrastructure_error"):
                    error_kind, reason = "tool_transport", "infrastructure_error"
                    error_msg = str(executed["infrastructure_error"])
                    break
            else:
                if not done and turns >= max_turns:
                    reason = "request_limit"
                elif success:
                    reason = "success"
                elif truncated:
                    reason = "environment_step_limit"
        finally:
            try:
                await client.post(
                    f"{sim_url.rstrip('/')}/sessions/{sid}/close", json={}
                )
            except Exception:
                pass
    finally:
        if owns_client:
            await client.aclose()

    outcome = {
        "episode_key": episode_key,
        "query_id": query_id,
        "pass_idx": pass_idx,
        "environment": conv.environment,
        "task": task,
        "init_state_id": opened.get("init_state_id"),
        "success": success,
        "truncated": truncated,
        "turns": turns,
        "termination_reason": "success" if success else reason,
        "error_kind": error_kind,
        "error": error_msg,
        "compaction_count": len(conv.compaction_events),
        "num_images": len(conv.all_images_by_id()),
        "elapsed_s": round(time.perf_counter() - started, 2),
    }

    if output_dir is not None:
        if save_trajectory:
            rollout_config = {
                "max_turns": max_turns,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k if resolved_provider == "sglang" else None,
                "thinking": thinking,
                "image_size": image_size,
                "max_actions_per_call": max_actions_per_call,
                "include_jaw_mm": include_jaw_mm,
                "compaction": (
                    {
                        "strategy": compaction.strategy,
                        "max_messages": compaction.max_messages,
                        "max_tokens": compaction.max_tokens,
                        "keep_messages": compaction.keep_messages,
                        "keep_recent_images": compaction.keep_recent_images,
                        "keep_recent_thinking": compaction.keep_recent_thinking,
                        "preserve_opening_images": compaction.preserve_opening_images,
                    }
                    if compaction is not None
                    else None
                ),
            }
            _save_episode_atomic(
                ep_dir=output_dir,
                episode_key=episode_key or output_dir.name,
                query_id=query_id or (episode_key or output_dir.name),
                pass_idx=pass_idx,
                conv=conv,
                outcome=outcome,
                model_name=model_name,
                provider=resolved_provider,
                model_url=active_model_url,
                rollout_config=rollout_config,
                usage_totals=usage_totals,
            )
        else:
            _atomic_write_json(output_dir / "outcome.json", outcome)

    return outcome


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-url",
        default=None,
        help="OpenAI-compatible /v1 URL (comma-separated for multi-replica routing)",
    )
    parser.add_argument(
        "--model-urls-file",
        type=Path,
        default=None,
        help="Optional newline-delimited file of /v1 URLs for sticky load balancing",
    )
    parser.add_argument("--model-name", default="policy")
    parser.add_argument(
        "--provider",
        choices=("auto", "sglang", "openai"),
        default="auto",
        help="API dialect: sglang (adds top_k/chat_template_kwargs), openai, or auto",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional Bearer API key (prefer --api-key-env to avoid shell history)",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable holding optional Bearer API key",
    )
    parser.add_argument("--sim-url", required=True, help="SimPoolServer HTTP base URL")
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Total trajectories to collect (cycles passes r0, r1... if > shard rows)",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="Optional intra-task shard '<index>/<total>' (e.g. 0/2 vs 1/2 across teachers)",
    )
    parser.add_argument(
        "--interleave-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Round-robin across tasks so every task is covered evenly (default: true)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip episodes already saved in output_dir/episodes/*/trajectory.json",
    )
    parser.add_argument(
        "--save-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save full message trajectories + PNGs per episode for SFT export",
    )
    parser.add_argument("--pass-offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable reasoning/thinking mode (default: true)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override per-environment default VLM display size (default: env-specific)",
    )
    parser.add_argument("--max-actions-per-call", type=int, default=3)
    parser.add_argument(
        "--include-jaw-mm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expose measured physical gripper jaw_mm (mm) in robot state feedback",
    )
    parser.add_argument(
        "--context-management",
        choices=("base", "dynamic"),
        default="dynamic",
        help="History compaction strategy: dynamic (RL-aligned default) or base",
    )
    parser.add_argument(
        "--compaction-max-tokens",
        type=int,
        default=None,
        help="Explicit token trigger for history compaction (default: 48000 for dynamic)",
    )
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--max-llm-retries", type=int, default=8)
    parser.add_argument("--max-infra-retries", type=int, default=3)
    parser.add_argument("--record-artifacts", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    model_urls = parse_model_urls(args.model_url, args.model_urls_file)
    api_key = args.api_key or (
        os.environ.get(args.api_key_env) if args.api_key_env else None
    )
    resolved_provider = detect_provider(
        args.provider,
        model_name=args.model_name,
        model_urls=model_urls,
        api_key=api_key,
    )
    shard_idx, num_shards = parse_shard_spec(args.shard)

    rows = [
        json.loads(line)
        for line in args.tasks_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    schedule = build_episode_schedule(
        rows,
        interleave_tasks=args.interleave_tasks,
        shard_index=shard_idx,
        num_shards=num_shards,
        target_count=args.target_count,
        limit=args.limit,
        pass_offset=args.pass_offset,
    )

    compaction_max_tokens = args.compaction_max_tokens
    compaction_max_messages: int | None = 80
    if args.context_management == "dynamic":
        compaction_max_messages = None
        if compaction_max_tokens is None:
            compaction_max_tokens = 48000
    compaction = CompactionConfig(
        strategy=args.context_management,
        max_messages=compaction_max_messages,
        max_tokens=compaction_max_tokens,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_episodes(args.output_dir) if args.resume else {}
    pending = [item for item in schedule if item["episode_key"] not in completed]

    logger.info(
        "Rollout starting: model=%s provider=%s endpoints=%d sim=%s concurrency=%d "
        "scheduled=%d resumed=%d pending=%d shard=%d/%d",
        args.model_name,
        resolved_provider,
        len(model_urls),
        args.sim_url,
        args.concurrency,
        len(schedule),
        len(schedule) - len(pending),
        len(pending),
        shard_idx,
        num_shards,
    )

    pool = EndpointPool(model_urls)
    manifest_path = args.output_dir / "manifest.jsonl"
    infra_err_path = args.output_dir / "infra_errors.jsonl"
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    stats = {
        "done": len(schedule) - len(pending),
        "solved": sum(
            1
            for item in schedule
            if item["episode_key"] in completed
            and completed[item["episode_key"]].get("success")
        ),
        "infra_errors": 0,
        "turns_sum": 0,
        "new_done": 0,
        "started_at": time.perf_counter(),
    }

    async def _run_all() -> list[dict[str, Any]]:
        limits = httpx.Limits(
            max_connections=max(64, args.concurrency * 4),
            max_keepalive_connections=max(32, args.concurrency * 2),
            keepalive_expiry=30.0,
        )
        async with httpx.AsyncClient(
            timeout=args.request_timeout_s,
            trust_env=False,
            limits=limits,
        ) as client:

            async def _one(item: dict[str, Any]) -> dict[str, Any] | None:
                ep_key = item["episode_key"]
                ep_dir = args.output_dir / "episodes" / ep_key
                async with sem:
                    chosen_url = pool.acquire()
                    try:
                        outcome = await run_episode(
                            model_url=chosen_url,
                            model_name=args.model_name,
                            sim_url=args.sim_url,
                            task=item["task"],
                            max_turns=args.max_turns,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            thinking=args.thinking,
                            compaction=compaction,
                            image_size=args.image_size,
                            max_actions_per_call=args.max_actions_per_call,
                            include_jaw_mm=args.include_jaw_mm,
                            record_artifacts=args.record_artifacts,
                            output_dir=ep_dir,
                            request_timeout_s=args.request_timeout_s,
                            api_key=api_key,
                            provider=resolved_provider,
                            max_llm_retries=args.max_llm_retries,
                            max_infra_retries=args.max_infra_retries,
                            save_trajectory=args.save_trajectory,
                            episode_key=ep_key,
                            query_id=item["query_id"],
                            pass_idx=item["pass_idx"],
                            client=client,
                            endpoint_pool=pool,
                        )
                    except Exception as exc:
                        err_rec = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "episode_key": ep_key,
                            "query_id": item["query_id"],
                            "endpoint": chosen_url,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                        async with write_lock:
                            stats["infra_errors"] += 1
                            with infra_err_path.open("a", encoding="utf-8") as fh:
                                fh.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                        logger.error(
                            "Episode %s failed with infra error (%s: %s); skipped so --resume can retry",
                            ep_key,
                            type(exc).__name__,
                            exc,
                        )
                        return None
                    finally:
                        pool.release(chosen_url)

                manifest_rec = {
                    "episode_key": ep_key,
                    "query_id": item["query_id"],
                    "pass_idx": item["pass_idx"],
                    "environment": outcome["environment"],
                    "success": bool(outcome["success"]),
                    "turns": int(outcome["turns"]),
                    "termination_reason": outcome["termination_reason"],
                    "error_kind": outcome["error_kind"],
                    "elapsed_s": outcome["elapsed_s"],
                    "trajectory_path": f"episodes/{ep_key}/trajectory.json",
                }
                async with write_lock:
                    with manifest_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(manifest_rec, ensure_ascii=False) + "\n")
                    completed[ep_key] = manifest_rec
                    stats["done"] += 1
                    stats["new_done"] += 1
                    stats["turns_sum"] += int(outcome["turns"])
                    if outcome["success"]:
                        stats["solved"] += 1
                    if stats["new_done"] % 10 == 0 or stats["done"] == len(schedule):
                        elapsed = max(1e-6, time.perf_counter() - stats["started_at"])
                        rate_hr = 3600.0 * stats["new_done"] / elapsed
                        logger.info(
                            "Progress %d/%d (solved=%d %.1f%%, avg_turns=%.1f, "
                            "throughput=%.1f ep/hr, infra_errors=%d)",
                            stats["done"],
                            len(schedule),
                            stats["solved"],
                            100.0 * stats["solved"] / max(stats["done"], 1),
                            stats["turns_sum"] / max(stats["new_done"], 1),
                            rate_hr,
                            stats["infra_errors"],
                        )
                return outcome

            raw_results = await asyncio.gather(*(_one(item) for item in pending))
            return [r for r in raw_results if r is not None]

    new_results = asyncio.run(_run_all())
    all_records = [
        completed[item["episode_key"]]
        for item in schedule
        if item["episode_key"] in completed
    ]
    solved = sum(bool(r.get("success")) for r in all_records)
    summary = {
        "model_name": args.model_name,
        "provider": resolved_provider,
        "scheduled": len(schedule),
        "episodes_completed": len(all_records),
        "episodes_new": len(new_results),
        "infra_errors": stats["infra_errors"],
        "solved": solved,
        "success_rate": solved / max(len(all_records), 1),
        "results": all_records,
    }
    _atomic_write_json(args.output_dir / "summary.json", summary)
    logger.info(
        "Rollout complete: %d/%d completed (%d solved, %.1f%%, %d infra errors)",
        len(all_records),
        len(schedule),
        solved,
        100.0 * summary["success_rate"],
        stats["infra_errors"],
    )


if __name__ == "__main__":
    main()

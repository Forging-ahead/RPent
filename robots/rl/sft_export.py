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

"""Export collected Harness v2 ``trajectory.json`` episodes to SFT datasets.

Supports three target formats:
- ``ms_swift`` (default): Canonical ``modelscope/ms-swift`` multimodal agent
  format (``{"tools": "<json-str>", "messages": [...], "images": [...]}``)
  designed for ``Qwen3.5-4B`` (``--template qwen3_5``, auto-selecting
  ``agent_template=qwen3_5``). Verified to render byte-identically to
  ``Qwen3.5-4B/chat_template.jinja`` used by SGLang/RLinf during RL.
- ``qwen_hf``: HuggingFace ``AutoTokenizer.apply_chat_template`` format with
  ``reasoning_content`` and ``dict`` ``tool_calls[*].function.arguments``.
- ``openai_messages``: Standard OpenAI chat-completions message format with
  JSON-string ``tool_calls[*].function.arguments``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any, Literal

from robots.rl.conversation import IMAGE_PLACEHOLDER_PREFIX

ExportFormat = Literal["ms_swift", "qwen_hf", "openai_messages"]
HistoryMode = Literal["full", "compacted", "auto"]


def _parse_tool_arguments(raw_args: Any) -> dict[str, Any] | None:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _estimate_message_tokens(messages: Sequence[dict[str, Any]]) -> int:
    chars = 0
    images = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    chars += len(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    images += 1
        rc = msg.get("reasoning_content")
        if isinstance(rc, str):
            chars += len(rc)
        for tc in msg.get("tool_calls") or ():
            chars += len(json.dumps(tc, ensure_ascii=False))
    # Conservative 4 chars/token + ~512 tokens/image estimate
    return (chars // 4) + (images * 512)


def _select_raw_messages(
    trajectory: dict[str, Any],
    *,
    history_mode: HistoryMode,
    max_seq_tokens: int | None,
) -> list[dict[str, Any]]:
    full_msgs = trajectory.get("full_messages") or []
    compacted_msgs = trajectory.get("compacted_messages") or full_msgs
    if history_mode == "compacted":
        return list(compacted_msgs)
    if history_mode == "full":
        return list(full_msgs)
    if max_seq_tokens is not None and _estimate_message_tokens(full_msgs) > max_seq_tokens:
        return list(compacted_msgs)
    return list(full_msgs)


def _trim_trailing_non_assistant(
    messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop trailing tool/user messages that have no subsequent assistant response."""
    end = len(messages)
    while end > 0 and messages[end - 1].get("role") != "assistant":
        end -= 1
    return list(messages[:end])


def _resolve_image_path(
    fid: str,
    *,
    image_map: dict[str, str],
    traj_dir: Path,
    image_root: Path | None,
) -> tuple[str, Path]:
    rel = image_map.get(fid) or f"images/{fid}.png"
    abs_path = (traj_dir / rel).resolve()
    if image_root is not None:
        try:
            display = str(abs_path.relative_to(image_root.resolve()))
        except ValueError:
            display = str(abs_path)
    else:
        display = str(abs_path)
    return display, abs_path


def _extract_placeholder_fid(url: str) -> str:
    if url.startswith(IMAGE_PLACEHOLDER_PREFIX):
        return url[len(IMAGE_PLACEHOLDER_PREFIX) :]
    raise ValueError(f"Unexpected image_url placeholder: {url!r}")


def _format_ms_swift_assistant_content(
    content: str | None, reasoning_content: str | None
) -> str:
    """Wrap assistant reasoning in ``<think>...</think>`` matching Qwen3.5 Jinja."""
    reasoning = (reasoning_content or "").strip()
    text = (content or "").strip()
    if reasoning:
        think_block = f"<think>\n{reasoning}\n</think>"
    else:
        think_block = "<think>\n\n</think>"
    if text:
        return f"{think_block}\n\n{text}"
    return f"{think_block}\n"


def _is_validation_retry_message(msg: dict[str, Any]) -> bool:
    return (
        msg.get("role") == "user"
        and isinstance(msg.get("content"), str)
        and msg["content"].startswith("Validation feedback:\n")
    )


def convert_trajectory(
    trajectory: dict[str, Any],
    *,
    traj_dir: Path,
    export_format: ExportFormat = "ms_swift",
    history_mode: HistoryMode = "full",
    max_seq_tokens: int | None = None,
    trim_trailing_non_assistant: bool = True,
    mask_retry_turns: bool = False,
    image_root: Path | None = None,
    check_images_exist: bool = True,
    include_metadata: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert one saved ``trajectory.json`` dict into a target SFT sample.

    Returns ``(sample_dict, drop_reason)``. Exactly one of the two is non-None.
    """
    raw_messages = _select_raw_messages(
        trajectory, history_mode=history_mode, max_seq_tokens=max_seq_tokens
    )
    if trim_trailing_non_assistant:
        raw_messages = _trim_trailing_non_assistant(raw_messages)
    if not raw_messages or not any(m.get("role") == "assistant" for m in raw_messages):
        return None, "no_assistant_turns"

    tools = trajectory.get("tools") or []
    image_map: dict[str, str] = trajectory.get("images") or {}
    images_out: list[str] = []

    if export_format == "ms_swift":
        ms_messages: list[dict[str, Any]] = []
        for idx, msg in enumerate(raw_messages):
            role = msg.get("role")
            if role == "system":
                ms_messages.append(
                    {"role": "system", "content": str(msg.get("content") or "")}
                )
            elif role == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    ms_messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    parts: list[str] = []
                    for part in content:
                        ptype = part.get("type")
                        if ptype == "text":
                            parts.append(str(part.get("text", "")))
                        elif ptype == "image_url":
                            fid = _extract_placeholder_fid(part["image_url"]["url"])
                            img_ref, abs_path = _resolve_image_path(
                                fid,
                                image_map=image_map,
                                traj_dir=traj_dir,
                                image_root=image_root,
                            )
                            if check_images_exist and not abs_path.is_file():
                                return None, f"missing_image:{fid}"
                            images_out.append(img_ref)
                            parts.append("<image>")
                        else:
                            return None, f"unsupported_content_part:{ptype}"
                    ms_messages.append({"role": "user", "content": "".join(parts)})
                else:
                    return None, "invalid_user_content"
            elif role == "assistant":
                is_masked_retry = (
                    mask_retry_turns
                    and idx + 1 < len(raw_messages)
                    and _is_validation_retry_message(raw_messages[idx + 1])
                )
                asst_text = _format_ms_swift_assistant_content(
                    msg.get("content"), msg.get("reasoning_content")
                )
                asst_entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": asst_text,
                }
                if is_masked_retry:
                    asst_entry["loss"] = False
                ms_messages.append(asst_entry)

                for tc in msg.get("tool_calls") or ():
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    parsed_args = _parse_tool_arguments(fn.get("arguments"))
                    if not name or parsed_args is None:
                        return None, "malformed_tool_call_arguments"
                    tc_entry: dict[str, Any] = {
                        "role": "tool_call",
                        "content": json.dumps(
                            {"name": name, "arguments": parsed_args},
                            ensure_ascii=False,
                        ),
                    }
                    if is_masked_retry:
                        tc_entry["loss"] = False
                    ms_messages.append(tc_entry)
            elif role == "tool":
                ms_messages.append(
                    {
                        "role": "tool_response",
                        "content": str(msg.get("content") or ""),
                    }
                )
            else:
                return None, f"unsupported_role:{role}"

        # Guard: exact 1-to-1 match between <image> placeholders and images list
        tag_count = sum(
            m["content"].count("<image>")
            for m in ms_messages
            if isinstance(m.get("content"), str)
        )
        if tag_count != len(images_out):
            return None, f"image_tag_mismatch:{tag_count}!={len(images_out)}"

        sample: dict[str, Any] = {
            # JSON string matches ms-swift Agent-support.md and prevents Arrow
            # struct schema conflicts when mixing heterogeneous environments.
            "tools": json.dumps(tools, ensure_ascii=False),
            "messages": ms_messages,
            "images": images_out,
        }
    else:
        converted_msgs: list[dict[str, Any]] = []
        for msg in raw_messages:
            role = msg.get("role")
            if role == "system":
                converted_msgs.append(
                    {"role": "system", "content": str(msg.get("content") or "")}
                )
            elif role == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    converted_msgs.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    new_parts: list[dict[str, Any]] = []
                    for part in content:
                        ptype = part.get("type")
                        if ptype == "text":
                            new_parts.append(
                                {"type": "text", "text": str(part.get("text", ""))}
                            )
                        elif ptype == "image_url":
                            fid = _extract_placeholder_fid(part["image_url"]["url"])
                            img_ref, abs_path = _resolve_image_path(
                                fid,
                                image_map=image_map,
                                traj_dir=traj_dir,
                                image_root=image_root,
                            )
                            if check_images_exist and not abs_path.is_file():
                                return None, f"missing_image:{fid}"
                            images_out.append(img_ref)
                            if export_format == "qwen_hf":
                                new_parts.append({"type": "image", "image": img_ref})
                            else:
                                new_parts.append(
                                    {"type": "image_url", "image_url": {"url": img_ref}}
                                )
                        else:
                            return None, f"unsupported_content_part:{ptype}"
                    converted_msgs.append({"role": "user", "content": new_parts})
                else:
                    return None, "invalid_user_content"
            elif role == "assistant":
                out_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                }
                if msg.get("reasoning_content") is not None:
                    out_msg["reasoning_content"] = str(msg["reasoning_content"])
                tcs = msg.get("tool_calls") or []
                if tcs:
                    formatted_tcs: list[dict[str, Any]] = []
                    for tc_idx, tc in enumerate(tcs):
                        fn = tc.get("function") or {}
                        name = fn.get("name")
                        parsed_args = _parse_tool_arguments(fn.get("arguments"))
                        if not name or parsed_args is None:
                            return None, "malformed_tool_call_arguments"
                        fn_args: Any = (
                            parsed_args
                            if export_format == "qwen_hf"
                            else json.dumps(parsed_args, ensure_ascii=False)
                        )
                        formatted_tcs.append(
                            {
                                "id": tc.get("id") or f"call_{tc_idx}",
                                "type": "function",
                                "function": {"name": name, "arguments": fn_args},
                            }
                        )
                    out_msg["tool_calls"] = formatted_tcs
                converted_msgs.append(out_msg)
            elif role == "tool":
                out_tool: dict[str, Any] = {
                    "role": "tool",
                    "content": str(msg.get("content") or ""),
                }
                if msg.get("tool_call_id"):
                    out_tool["tool_call_id"] = str(msg["tool_call_id"])
                converted_msgs.append(out_tool)
            else:
                return None, f"unsupported_role:{role}"

        sample = {
            "tools": tools,
            "messages": converted_msgs,
            "images": images_out,
        }

    if include_metadata:
        outcome = trajectory.get("outcome") or {}
        sample["metadata"] = {
            "episode_key": trajectory.get("episode_key"),
            "query_id": trajectory.get("query_id"),
            "environment": trajectory.get("environment"),
            "teacher_model": (trajectory.get("model") or {}).get("name"),
            "success": bool(outcome.get("success", False)),
            "turns": int(outcome.get("turns", 0)),
            "termination_reason": outcome.get("termination_reason"),
            "error_kind": outcome.get("error_kind"),
        }

    return sample, None


def render_ms_swift_row_with_qwen3_5_template(
    sample: dict[str, Any],
    *,
    ms_swift_repo: Path | None = None,
) -> tuple[str, list[tuple[str, float]]]:
    """Render an ``ms_swift`` row via ``ms-swift``'s actual ``Qwen3_5Template`` logic.

    Returns ``(rendered_text, segments_with_loss_scale)`` where ``<image>`` is
    expanded to ``<|vision_start|><|image_pad|><|vision_end|>`` so it can be
    compared byte-for-byte against HuggingFace ``Qwen3.5-4B/chat_template.jinja``.
    """
    if ms_swift_repo is not None and str(ms_swift_repo) not in sys.path:
        sys.path.insert(0, str(ms_swift_repo))

    from swift.agent_template import agent_template_map
    from swift.loss_scale import get_loss_scale
    from swift.template.base import Template
    from swift.template.template_inputs import StdTemplateInputs
    from swift.template.templates.qwen import Qwen3_5Template, QwenTemplateMeta
    from swift.template.utils import ContextType

    inputs = StdTemplateInputs.from_dict(sample)
    agent = agent_template_map["qwen3_5"]()
    meta = QwenTemplateMeta(
        "qwen3_5",
        template_cls=Qwen3_5Template,
        default_system=None,
        thinking_prefix="<think>\n",
        non_thinking_prefix="<think>\n\n</think>\n\n",
        default_enable_thinking=True,
        agent_template="qwen3_5",
        is_thinking=True,
    )
    agent.template_meta = meta

    # Minimal lightweight stand-in reusing ms-swift's exact methods without GPU/weights
    class _LightweightQwen3_5(Qwen3_5Template):
        def __init__(self) -> None:
            self.template_meta = meta
            self.template_backend = "swift"
            self.use_chat_template = True
            self.mode = "train"
            self.task_type = "causal_lm"
            self.add_non_thinking_prefix = True
            self.preserve_thinking = None
            self.enable_thinking = True
            self.response_prefix = None
            self.chat_template_kwargs = {}
            self._agent_template = "qwen3_5"
            self._agent_template_cache = {"qwen3_5": agent}
            self._loss_scale = "default"
            self._loss_scale_cache = {"default": get_loss_scale("default")}
            self.is_binary_loss_scale = True

    tmpl = _LightweightQwen3_5()
    tmpl._preprocess_tools(inputs)
    tmpl._swift_prepare_inputs(inputs)
    res_ctx, loss_scales, _ = Template._swift_encode(tmpl, inputs)

    rendered_parts: list[str] = []
    segments: list[tuple[str, float]] = []
    for ctx, scale in zip(res_ctx, loss_scales, strict=True):
        text = str(ctx).replace(
            "<image>", "<|vision_start|><|image_pad|><|vision_end|>"
        )
        rendered_parts.append(text)
        segments.append((text, float(scale)))
    # Ensure standard suffix marker is present in rendered output
    assert ContextType.RESPONSE is not None
    return "".join(rendered_parts), segments


def discover_trajectory_files(input_dirs: Sequence[Path]) -> list[Path]:
    """Find all ``trajectory.json`` files under one or more collector output dirs."""
    found: list[Path] = []
    for root in input_dirs:
        if root.is_file() and root.name == "trajectory.json":
            found.append(root)
        elif root.is_dir():
            if (root / "trajectory.json").is_file():
                found.append(root / "trajectory.json")
            else:
                found.extend(sorted(root.rglob("trajectory.json")))
    return sorted(dict.fromkeys(found))


def verify_exported_samples(
    samples: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    tokenizer_path: Path | None = None,
    verify_ms_swift: bool = False,
    ms_swift_repo: Path | None = None,
    verify_limit: int | None = None,
) -> dict[str, Any]:
    """Verify exported samples against Qwen3.5 Jinja tokenizer and/or ``ms-swift``.

    ``samples`` is a sequence of ``(exported_row, qwen_hf_row)`` pairs.
    """
    subset = samples[:verify_limit] if verify_limit is not None else samples
    tokenizer = None
    if tokenizer_path is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), trust_remote_code=True
        )

    token_lengths: list[int] = []
    exact_matches = 0
    retry_diffs = 0
    checked = 0

    for exported_row, qwen_hf_row in subset:
        checked += 1
        expected_images = len(exported_row.get("images") or ())
        hf_text: str | None = None

        if tokenizer is not None:
            hf_text = tokenizer.apply_chat_template(
                qwen_hf_row["messages"],
                tools=qwen_hf_row["tools"],
                tokenize=False,
                add_generation_prompt=False,
            )
            pad_count = hf_text.count("<|vision_start|><|image_pad|><|vision_end|>")
            if pad_count != expected_images:
                raise AssertionError(
                    f"HF chat_template image pad mismatch: {pad_count} != {expected_images}"
                )
            if "<tool_call>" in hf_text and "<function=" not in hf_text:
                raise AssertionError("Rendered <tool_call> missing <function=...> tag")
            # Estimate total tokens (text tokens + conservative 256 vision tokens/image)
            text_toks = len(tokenizer.encode(hf_text, add_special_tokens=False))
            token_lengths.append(text_toks)

        if verify_ms_swift:
            ms_text, segments = render_ms_swift_row_with_qwen3_5_template(
                exported_row, ms_swift_repo=ms_swift_repo
            )
            ms_pad = ms_text.count("<|vision_start|><|image_pad|><|vision_end|>")
            if ms_pad != expected_images:
                raise AssertionError(
                    f"ms-swift image pad mismatch: {ms_pad} != {expected_images}"
                )
            active_segments = [text for text, scale in segments if scale > 0.0]
            if not active_segments:
                raise AssertionError("ms-swift produced zero supervised assistant tokens")
            if any("<|image_pad|>" in text for text in active_segments):
                raise AssertionError("ms-swift leaked image tokens into supervised labels")

            if hf_text is not None:
                has_validation_retry = any(
                    _is_validation_retry_message(m)
                    for m in qwen_hf_row["messages"]
                )
                if ms_text == hf_text:
                    exact_matches += 1
                elif has_validation_retry:
                    # Expected difference: HF Jinja drops <think> prior to
                    # non-<tool_response> Validation feedback user turns, while
                    # ms-swift SFT (preserve_thinking=True) keeps all <think>.
                    retry_diffs += 1
                else:
                    # Locate first differing character for actionable diagnostics
                    diff_idx = next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(ms_text, hf_text))
                            if a != b
                        ),
                        min(len(ms_text), len(hf_text)),
                    )
                    raise AssertionError(
                        f"ms-swift vs Qwen3.5 HF Jinja mismatch at char {diff_idx}:\n"
                        f"  ms-swift: {ms_text[max(0, diff_idx - 80):diff_idx + 120]!r}\n"
                        f"  hf-jinja: {hf_text[max(0, diff_idx - 80):diff_idx + 120]!r}"
                    )

    token_stats: dict[str, int] = {}
    if token_lengths:
        ordered = sorted(token_lengths)
        token_stats = {
            "text_tokens_min": ordered[0],
            "text_tokens_p50": ordered[len(ordered) // 2],
            "text_tokens_p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "text_tokens_max": ordered[-1],
        }
    return {
        "checked": checked,
        "exact_ms_swift_hf_matches": exact_matches,
        "expected_retry_thinking_diffs": retry_diffs,
        **token_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more rollout output directories (or trajectory.json files)",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("ms_swift", "qwen_hf", "openai_messages"),
        default="ms_swift",
        help="Target SFT schema (default: ms_swift for Qwen3.5-4B)",
    )
    parser.add_argument(
        "--history-mode",
        choices=("full", "compacted", "auto"),
        default="full",
        help="Export lossless full_messages, compacted_messages, or auto-fallback",
    )
    parser.add_argument("--max-seq-tokens", type=int, default=None)
    parser.add_argument("--only-success", action="store_true")
    parser.add_argument("--drop-truncated", action="store_true")
    parser.add_argument("--drop-malformed-tool-calls", action="store_true")
    parser.add_argument(
        "--trim-trailing-non-assistant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim trailing tool_response/user turns with no assistant follow-up",
    )
    parser.add_argument(
        "--mask-retry-turns",
        action="store_true",
        help="(ms_swift) Set loss=false on assistant turns that triggered validation retry",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional root directory to store relative image paths (for ROOT_IMAGE_DIR)",
    )
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument(
        "--verify-tokenizer",
        type=Path,
        default=None,
        help="Path to Qwen3.5 tokenizer/model dir to verify HF chat_template rendering",
    )
    parser.add_argument(
        "--verify-ms-swift",
        action="store_true",
        help="Verify exported ms_swift rows with ms-swift Qwen3_5Template",
    )
    parser.add_argument(
        "--ms-swift-repo",
        type=Path,
        default=None,
        help="Optional path to cloned ms-swift repo if not installed in venv",
    )
    parser.add_argument("--verify-limit", type=int, default=None)
    args = parser.parse_args()

    traj_files = discover_trajectory_files(args.input_dirs)
    drop_counts: Counter[str] = Counter()
    env_counts: Counter[str] = Counter()
    teacher_counts: Counter[str] = Counter()
    success_count = 0

    exported_rows: list[dict[str, Any]] = []
    verify_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for traj_path in traj_files:
        try:
            traj = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception:
            drop_counts["corrupt_json"] += 1
            continue
        outcome = traj.get("outcome") or {}
        if args.only_success and not outcome.get("success"):
            drop_counts["not_success"] += 1
            continue
        if args.drop_truncated and outcome.get("termination_reason") == "response_truncated":
            drop_counts["response_truncated"] += 1
            continue
        if args.drop_malformed_tool_calls and outcome.get("error_kind") in {
            "tool_name",
            "tool_arguments",
        }:
            drop_counts["malformed_tool_call"] += 1
            continue

        sample, drop_reason = convert_trajectory(
            traj,
            traj_dir=traj_path.parent,
            export_format=args.format,
            history_mode=args.history_mode,
            max_seq_tokens=args.max_seq_tokens,
            trim_trailing_non_assistant=args.trim_trailing_non_assistant,
            mask_retry_turns=args.mask_retry_turns,
            image_root=args.image_root,
            check_images_exist=True,
            include_metadata=args.include_metadata,
        )
        if sample is None:
            drop_counts[drop_reason or "unknown"] += 1
            continue

        if args.verify_tokenizer is not None or args.verify_ms_swift:
            qwen_hf_sample, hf_drop = convert_trajectory(
                traj,
                traj_dir=traj_path.parent,
                export_format="qwen_hf",
                history_mode=args.history_mode,
                max_seq_tokens=args.max_seq_tokens,
                trim_trailing_non_assistant=args.trim_trailing_non_assistant,
                image_root=args.image_root,
                check_images_exist=True,
            )
            if qwen_hf_sample is None:
                drop_counts[f"hf_verify:{hf_drop}"] += 1
                continue
            verify_pairs.append((sample, qwen_hf_sample))

        exported_rows.append(sample)
        env_counts[str(traj.get("environment") or "unknown")] += 1
        teacher_counts[str((traj.get("model") or {}).get("name") or "unknown")] += 1
        if outcome.get("success"):
            success_count += 1

    verify_summary: dict[str, Any] = {}
    if verify_pairs:
        verify_summary = verify_exported_samples(
            verify_pairs,
            tokenizer_path=args.verify_tokenizer,
            verify_ms_swift=args.verify_ms_swift and args.format == "ms_swift",
            ms_swift_repo=args.ms_swift_repo,
            verify_limit=args.verify_limit,
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with tmp_out.open("w", encoding="utf-8") as fh:
        for row in exported_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_out.replace(args.output_jsonl)

    stats = {
        "format": args.format,
        "history_mode": args.history_mode,
        "input_trajectories": len(traj_files),
        "exported": len(exported_rows),
        "exported_success": success_count,
        "exported_success_rate": success_count / max(len(exported_rows), 1),
        "by_environment": dict(env_counts),
        "by_teacher": dict(teacher_counts),
        "dropped": dict(drop_counts),
        "verification": verify_summary,
    }
    stats_path = args.output_jsonl.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

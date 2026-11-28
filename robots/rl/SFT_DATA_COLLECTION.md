# Harness v2 SFT Data Collection & `ms-swift` Distillation Guide

This guide explains how to collect large-scale, RL-aligned multimodal manipulation trajectories using **Harness v2** (`robots/rl/rollout.py`) and export verified datasets ready for **`ms-swift` SFT** (`Qwen3.5-4B` or any Qwen3.5/Qwen3.8 model) via `robots/rl/sft_export.py`.

Every setting matches RLinf Harness v2 RL training (`max_turns=50`, `max_tokens=16384`, `context_management=dynamic`, `compaction_max_tokens=48000`, `include_jaw_mm=True`, per-env default resolutions, step-0 opening images, and `execute_actions(..., observe=...)`) so SFT checkpoints transfer to RL with **zero train/inference template mismatch**.

---

## 1. End-to-End Pipeline Overview

```text
  Teacher Models (Parallel)                 Dedicated Sim + SAM3 Nodes
┌─────────────────────────────────┐      ┌──────────────────────────────┐
│ Local Qwen3.8-27B (6× 8xH100)  │      │ Sim Node A (8× H100, 208 slots)│
│  24× TP=2 SGLang replicas       │◄────►│  64 LIBERO + 48 RoboTwin      │
│  Continuous batching + Radix    │      │  32 RoboLab + 64 MetaWorld    │
└─────────────────────────────────┘      │  + 8 SAM3 (port 28520)        │
                                         └──────────────────────────────┘
┌─────────────────────────────────┐      ┌──────────────────────────────┐
│ Remote API (deepseek-v4.1-flash)│      │ Sim Node B (8× H100, 208 slots)│
│  Via jump-host HTTPS reverse    │◄────►│  64 LIBERO + 48 RoboTwin      │
│  proxy (air-gapped VPC safe)    │      │  32 RoboLab + 64 MetaWorld    │
└─────────────────────────────────┘      │  + 8 SAM3 (port 28520)        │
                                         └──────────────────────────────┘
                  │                                     │
                  ▼   python -m robots.rl.rollout       ▼
         Per-Episode Atomic Storage (--resume safe at any time)
         <output_dir>/
           ├── manifest.jsonl
           ├── infra_errors.jsonl
           ├── summary.json
           └── episodes/<episode_key>/
                 ├── trajectory.json   (full_messages, compacted_messages, tools, outcome)
                 ├── outcome.json
                 └── images/<fid>.png  (deduplicated content-addressed PNGs)
                  │
                  ▼   python -m robots.rl.sft_export --format ms_swift
         Verified ms-swift JSONL (byte-identical to Qwen3.5-4B/chat_template.jinja)
                  │
                  ▼   swift sft --model Qwen3.5-4B --template qwen3_5 ...
```

---

## 2. Step 1 — Start Simulator + SAM3 Pools

Copy `tools/rl/sim_services_4env_sft_h100.example.yaml`, adjust paths, and start on each 8× 80 GB H100 simulator node:

```bash
ulimit -n 1048576
PYTHONPATH=/path/to/RPent \
  python /path/to/RPent/tools/rl/serve.py \
  --config /path/to/sim_services_4env_sft.yaml
```

Verify all 4 environment pools are healthy (`workers_idle == workers_target`):

```bash
for port in 28561 28562 28564 28565; do
  echo "=== :$port ===" && curl -s "http://127.0.0.1:${port}/health" && echo
done
```

> **Pre-flight reminders**: See [`robots/rl/README.md`](./README.md#pre-flight-checklist-for-a-dedicated-5-env-simulator-node) for 8-GPU Vulkan verification, offline `/root/.nvidia-omniverse/config/omniverse.toml`, and native `LD_LIBRARY_PATH`.

---

## 3. Step 2 — Serve Teacher Models

### Option A: Local `Qwen3.8-27B` SGLang Continuous-Batching Replicas (6× 8× H100 = 24× `TP=2`)

Serving `Qwen3.8-27B` with **`TP=2`** gives 4 independent replicas per 8-GPU node (24 replicas across 6 nodes) with generous KV/RadixCache headroom for 64k multimodal contexts. Launch 4 replicas per node (GPU pairs `0,1`, `2,3`, `4,5`, `6,7` on ports `30000..30003`):

```bash
export LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
export SGLANG_MM_BUFFER_SIZE_MB=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for i in 0 1 2 3; do
  g0=$((i * 2))
  g1=$((i * 2 + 1))
  port=$((30000 + i))
  CUDA_VISIBLE_DEVICES="${g0},${g1}" \
  nohup /opt/venv/reason/bin/python -m sglang.launch_server \
    --model-path /path/to/models/Qwen3.8-27B \
    --served-model-name Qwen3.8-27B \
    --host 0.0.0.0 --port "${port}" \
    --tp 2 \
    --context-length 65536 \
    --mem-fraction-static 0.85 \
    --max-running-requests 32 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --attention-backend triton \
    --trust-remote-code \
    > "/path/to/logs/sglang_$(hostname)_${port}.log" 2>&1 < /dev/null &
done
```

Put all 24 endpoints into a newline-delimited file (`qwen27b_endpoints.txt`):

```text
http://<node1_ip>:30000/v1
http://<node1_ip>:30001/v1
...
http://<node6_ip>:30003/v1
```

`robots/rl/rollout.py` uses **`EndpointPool` sticky least-loaded routing**: all turns of one episode stay pinned to the same SGLang replica (maximizing RadixCache prefix hits), while launching high `--concurrency` (e.g., `64 + 48 + 32 + 64 = 208` concurrent episodes across the 4 envs) feeds ~8–9 batched multimodal requests into every replica simultaneously.

### Option B: Remote OpenAI-Compatible API (`deepseek-v4.1-flash`) on Air-Gapped Clusters

When GPU nodes cannot reach the public internet directly, run a minimal `nginx` HTTPS reverse proxy on an internet-connected jump host restricted strictly to your VPC subnet:

```nginx
server {
    listen <JUMP_VPC_IP>:18080;
    allow <CLUSTER_VPC_CIDR>;
    deny all;

    client_max_body_size 64m;
    proxy_connect_timeout 60s;
    proxy_send_timeout 600s;
    proxy_read_timeout 600s;

    location /maas/v1/ {
        proxy_pass https://cloud.infini-ai.com/maas/v1/;
        proxy_ssl_server_name on;
        proxy_http_version 1.1;
        proxy_set_header Host cloud.infini-ai.com;
        proxy_set_header Connection "";
        proxy_set_header Authorization $http_authorization;
    }
}
```

Pass your API key **only via environment variable** (`export DEEPSEEK_API_KEY=...`) — never commit keys to configs or scripts.

---

## 4. Step 3 — Run Resumable Multi-Env SFT Collection

Prepare task JSONLs (for 2,000 unique seeds across 2 teachers on RoboLab's 120 tasks, generate `seeds 0-17` = 2,160 rows; MetaWorld `mt50.jsonl` has 50 tasks × 50 seeds = 2,500 rows; RoboTwin has 50 tasks × 40 seeds = 2,000 rows; LIBERO has 30 tasks × 50 states = 1,500 rows):

```bash
# Example: expand RoboLab to 120 tasks x 18 seeds = 2,160 unique rows
python /path/to/RLinf/examples/agent/rpent/prepare_tasks.py \
  --robolab-catalog /path/to/RPent/robots/robolab/task_catalog.json \
  --seeds 0-17 \
  --output-dir /path/to/data/robolab
```

### Launching Local `Qwen3.8-27B` Collectors (`--shard 0/2`, Full Sim Concurrency)

```bash
# Run one collector process per environment in parallel (total concurrency = 64+48+32+64 = 208)
PYTHONPATH=/path/to/RPent python -m robots.rl.rollout \
  --model-urls-file /path/to/qwen27b_endpoints.txt \
  --model-name Qwen3.8-27B \
  --provider sglang \
  --sim-url http://<SIM_NODE_A_IP>:28561 \
  --tasks-jsonl /path/to/data/libero/train.jsonl \
  --output-dir /path/to/sft_data/qwen3.8-27b/libero \
  --shard 0/2 \
  --interleave-tasks \
  --target-count 1000 \
  --concurrency 64 \
  --max-turns 50 \
  --max-tokens 16384 \
  --context-management dynamic \
  --compaction-max-tokens 44000 \
  --include-jaw-mm \
  --request-timeout-s 1200 \
  --resume
```

Repeat with:
- `robotwin`: `--sim-url http://<SIM_NODE_A_IP>:28562 --tasks-jsonl .../robotwin/train.jsonl --concurrency 16`
- `robolab`: `--sim-url http://<SIM_NODE_A_IP>:28564 --tasks-jsonl .../robolab/train.jsonl --concurrency 16`
- `metaworld`: `--sim-url http://<SIM_NODE_A_IP>:28565 --tasks-jsonl .../metaworld/mt50.jsonl --concurrency 64`

### Launching Remote `deepseek-v4.1-flash` Collectors (`--shard 1/2`, 128 Total Concurrency)

```bash
export DEEPSEEK_API_KEY="your-api-key"

# 32 concurrency x 4 environments = 128 total concurrent API requests
PYTHONPATH=/path/to/RPent python -m robots.rl.rollout \
  --model-url http://<JUMP_VPC_IP>:18080/maas/v1 \
  --model-name deepseek-v4.1-flash \
  --provider openai \
  --api-key-env DEEPSEEK_API_KEY \
  --sim-url http://<SIM_NODE_B_IP>:28561 \
  --tasks-jsonl /path/to/data/libero/train.jsonl \
  --output-dir /path/to/sft_data/deepseek-v4.1-flash/libero \
  --shard 1/2 \
  --interleave-tasks \
  --target-count 1000 \
  --concurrency 32 \
  --max-turns 50 \
  --max-tokens 16384 \
  --temperature 0.7 \
  --context-management dynamic \
  --compaction-max-tokens 48000 \
  --include-jaw-mm \
  --max-llm-retries 8 \
  --request-timeout-s 600 \
  --resume
```

### Interrupting, Resuming, and Sharding Across Teammates

- **Safe interruption at any time**: Every finished episode writes its deduplicated PNGs and atomically replaces `episodes/<episode_key>/trajectory.json` (`tmp` → `os.replace`) plus appends one line to `manifest.jsonl`. Re-running the exact same command with `--resume` (default) skips all completed episodes and retries any transient `infra_errors.jsonl` failures.
- **Disjoint multi-teacher / multi-contributor sharding**: `--shard <i>/<N>` shards *within each task* before round-robin interleaving across tasks (`--interleave-tasks`), so every shard covers 100% of tasks with zero seed overlap. For example, 4 teammates contributing to the same environment can run `--shard 0/4`, `1/4`, `2/4`, `3/4` and merge their output directories during export.

---

## 5. Step 4 — Export & Verify `ms-swift` SFT Dataset for `Qwen3.5-4B`

Run `robots/rl/sft_export.py` to merge trajectories from all environments and teachers into a verified `ms-swift` JSONL:

```bash
# Export successful trajectories (recommended for distillation SFT)
PYTHONPATH=/path/to/RPent python -m robots.rl.sft_export \
  --input-dirs \
    /path/to/sft_data/qwen3.8-27b \
    /path/to/sft_data/deepseek-v4.1-flash \
  --output-jsonl /path/to/sft_exports/rpent_4env_ms_swift_success.jsonl \
  --format ms_swift \
  --history-mode full \
  --only-success \
  --drop-truncated \
  --drop-malformed-tool-calls \
  --verify-tokenizer /path/to/models/Qwen3.5-4B \
  --verify-ms-swift

# Or export all trajectories (including failed episodes, e.g. for analysis / DPO / filtering)
PYTHONPATH=/path/to/RPent python -m robots.rl.sft_export \
  --input-dirs /path/to/sft_data/qwen3.8-27b /path/to/sft_data/deepseek-v4.1-flash \
  --output-jsonl /path/to/sft_exports/rpent_4env_ms_swift_all.jsonl \
  --format ms_swift \
  --history-mode full \
  --verify-tokenizer /path/to/models/Qwen3.5-4B \
  --verify-ms-swift
```

### Exact `ms-swift` Format Specification (`--format ms_swift`)

Each exported line follows `ms-swift`'s canonical multimodal agent schema (aligned with `swift/template/templates/qwen.py` `Qwen3_5Template` and `swift/agent_template/qwen3_coder.py` `Qwen3_5AgentTemplate`):

```json
{
  "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"view_env\", ...}}, ...]",
  "images": [
    "/path/to/episodes/libero_pro-task0-state0__r0/images/a1b2c3.png",
    "/path/to/episodes/libero_pro-task0-state0__r0/images/d4e5f6.png"
  ],
  "messages": [
    {"role": "system", "content": "<Harness v2 system prompt>"},
    {"role": "user", "content": "<task + initial_obs>\n<observation step=\"0\" camera=\"agentview\" file_id=\"a1b2c3\"><image></observation>..."},
    {"role": "assistant", "content": "<think>\n<step-by-step reasoning>\n</think>\n"},
    {"role": "tool_call", "content": "{\"name\": \"execute_actions\", \"arguments\": {\"actions\": [...], \"observe\": \"wrist\"}}"},
    {"role": "tool_response", "content": "[\"{...}\",\"See file d4e5f6.\"]"},
    {"role": "user", "content": "<tool_response>\n<tool_result tool_name=\"execute_actions\" tool_call_id=\"call_1\" file_id=\"d4e5f6\"><image></tool_result>\n</tool_response>"},
    {"role": "assistant", "content": "<think>\n<next reasoning>\n</think>\n"},
    {"role": "tool_call", "content": "{\"name\": \"execute_actions\", \"arguments\": {\"actions\": [...]}}"}
  ]
}
```

### Critical Qwen3.5-4B / `ms-swift` Alignment Details (Why SFT & RL Match Byte-for-Byte)

1. **Inline `<think>...</think>` inside `assistant.content`**: `ms-swift`'s `RowPreprocessor._check_messages()` strips every message key outside `{'role', 'content', 'loss', 'loss_scale'}` (discarding any separate `reasoning_content` key!). Wrapping reasoning as `<think>\n...\n</think>\n` inside `assistant.content` preserves full chain-of-thought through `ms-swift` preprocessing AND normalizes identically to `Qwen3.5-4B/chat_template.jinja` inside `Qwen3_5Template._swift_prepare_inputs()`.
2. **`"tools"` as a JSON string**: Our 4 environments have distinct `execute_actions` schemas (single-arm LIBERO/RoboLab, dual-arm RoboTwin with `arm`, 4-DoF MetaWorld without `orientation_rpy`). Serializing `"tools"` as a JSON string (per `ms-swift`'s `Agent-support.md`) prevents HuggingFace Arrow from injecting `null` struct fields or raising `ArrowInvalid` when combining datasets across environments.
3. **Canonical `tool_call` + `tool_response` roles**: `Qwen3_5AgentTemplate` converts `{"role": "tool_call", "content": "{\"name\": ..., \"arguments\": {...}}"}` into exact `<tool_call>\n<function=...>\n<parameter=...>\n...\n</parameter>\n</function>\n</tool_call>` XML, and `HermesAgentTemplate._format_tool_user_followup` merges `tool_response` + our `<tool_response><tool_result ...><image></tool_result></tool_response>` user message into exact `<|im_start|>user\n<tool_response>...<|im_end|>\n<|im_start|>user\n<tool_response>...<|im_end|>\n` turns — verified **100% byte-for-byte identical** to `Qwen3.5-4B/chat_template.jinja`!
4. **Trimming trailing non-assistant turns (`--trim-trailing-non-assistant`, default on)**: When an episode ends with `success=True` after `execute_actions`, the simulator returns a final `tool` result with no subsequent assistant turn. Trimming trailing `tool`/`user` turns prevents `ms-swift` from appending a dangling `<|im_start|>assistant\n<think>\n` with no target tokens.
5. **Other export formats**: Pass `--format qwen_hf` for native HuggingFace `apply_chat_template` training frameworks or `--format openai_messages` for standard OpenAI JSONL archives.

---

## 6. Step 5 — Training `Qwen3.5-4B` with `ms-swift`

Use `--template qwen3_5` (automatically selects `agent_template=qwen3_5`) and **`--loss_scale default`** (NEVER `last_round`, because `last_round` strips multi-turn `<think>` blocks and drops supervision on intermediate tool-calling turns!):

```bash
NPROC_PER_NODE=8 \
swift sft \
  --model /path/to/models/Qwen3.5-4B \
  --template qwen3_5 \
  --dataset /path/to/sft_exports/rpent_4env_ms_swift_success.jsonl \
  --train_type full \
  --torch_dtype bfloat16 \
  --max_length 65536 \
  --loss_scale default \
  --preserve_thinking true \
  --freeze_vit true \
  --freeze_aligner true \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --deepspeed zero3 \
  --output_dir /path/to/checkpoints/qwen3.5-4b-rpent-4env-sft
```

> **Note on `--image-root` / `ROOT_IMAGE_DIR`**: If moving exported JSONL + images across clusters, export with `--image-root /path/to/sft_data` and launch `ms-swift` with `ROOT_IMAGE_DIR=/new/path/to/sft_data`.

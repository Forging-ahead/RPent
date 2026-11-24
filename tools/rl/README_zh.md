# 机器人 Episode

[Installation, rollout and toolkit service](../../docs/source-zh/rst_source/simulators/rl.rst)

## ManiSkill PickCube

在已安装 RPent episode 依赖的独立 Python 环境中安装 `requirements/rl-maniskill.txt`。本地 `tools/rl/run.sh` 与远端 `robots.rl.remote_manager` 均使用这个 Python。远端入口使用现有的 `--simulator maniskill --gpu-slots GPU:1 --sam3-endpoint URL`。

任务参数为 `--simulator maniskill --task-name pick_cube --seed 0`。原生 `PickCube-v1` 使用 Panda 和固定 `agentview` 相机。`--max-episode-steps` 同时设置 Gym 时间上限和 RPent 物理步上限。

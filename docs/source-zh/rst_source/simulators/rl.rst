机器人 Episode
==============

用同一套 episode 在 LIBERO、RoboTwin、RoboCasa、RoboDojo 和 ManiSkill 中运行模型。ManiSkill 复用机械臂的公共动作接口，读取任务自身的策略相机。模型通过 ``execute_actions`` 执行动作，通过 ``view_env`` 获取策略相机图像。

安装仿真环境
------------

准备配套版本的 RPent 和 RLinf。在运行仿真的主机上执行对应的安装命令。安装脚本创建独立虚拟环境，并引用该 RLinf checkout 中的仿真适配器。

.. code-block:: bash

   bash tools/rl/install_simulator.sh libero   .venv-libero   "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robotwin .venv-robotwin "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robocasa .venv-robocasa "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh robodojo .venv-robodojo "$RLINF_ROOT"
   bash tools/rl/install_simulator.sh maniskill .venv-maniskill "$RLINF_ROOT"

.. list-table::
   :header-rows: 1

   * - 仿真器
     - 主机上的依赖和资产
   * - LIBERO
     - Python 3.12、EGL 和 MuJoCo。设置 ``LIBERO_TYPE=pro`` 与 ``LIBERO_CONFIG_PATH``，配置中给出资产、BDDL 和原生初态文件的位置。见 :doc:`libero`。
   * - RoboTwin
     - Python 3.11、NVIDIA Vulkan 和与 Torch 匹配的 CUDA 工具链，用于构建 cuRobo。在环境中运行 ``robotwin-download-assets``，将快照根目录设为 ``ROBOTWIN_ASSETS_PATH``。见 :doc:`robotwin`。
   * - RoboCasa
     - Python 3.11 和 EGL。在环境中运行 ``python -m robocasa.scripts.download_kitchen_assets --type all``。见 :doc:`robocasa`。
   * - RoboDojo
     - Python 3.11、Isaac Sim 5.1 和 RTX GPU。将 RoboDojo 资产和评测布局放在 ``ROBODOJO_ASSETS_ROOT``。见 :doc:`robodojo`。
   * - ManiSkill
     - Python 3.12、Linux 和支持 Vulkan 渲染的 NVIDIA GPU。已有资产时，将 ``MS_ASSET_DIR`` 指向含有 ``data/`` 的资产根目录。首次安装时，将该变量设为新目录，在仿真环境中执行 ``python -m mani_skill.utils.download_asset all -y``。对已有资产执行 ``-y`` 会替换对应文件。

准备兼容 OpenAI API 的模型服务和 SAM3 服务。在负责 SAM3 的 Python 环境中安装 ``sam3`` extra，把 ``SAM3_CHECKPOINT_PATH`` 设为本机权重文件。用分配给 SAM3 的 GPU 启动服务。

.. code-block:: bash

   "$SAM3_PYTHON" -m rpent.robots.components.sam3_server \
     --host 127.0.0.1 --port 28424 --cuda-device 1

运行一条轨迹
------------

激活仿真器的虚拟环境。下面用本机的 SAM3 和已经启动的模型服务运行 LIBERO。

.. code-block:: bash

   source .venv-libero/bin/activate
   export RPENT_MODEL=openai-chat:Qwen3.5-4B
   export RPENT_BASE_URL=http://127.0.0.1:30000/v1
   export RPENT_SAM3_ENDPOINT=http://127.0.0.1:28424
   export RPENT_OUTPUT_DIR="$PWD/runs/libero-first"
   bash tools/rl/run.sh --simulator libero --suite libero_spatial --task 0 --seed 0

RoboTwin、RoboCasa 和 RoboDojo 使用 ``--simulator`` 和 ``--task-name``。RoboDojo 可用 ``--layout`` 指定评测布局。输出目录保存图像、实测状态、工具回执和原生终局。批量采集用 ``tools/rl/run_batch.py`` 读取 JSON 任务清单，逐条封存并支持续跑。

ManiSkill 直接接受注册的 Gymnasium 任务 ID。例如，运行双臂任务可用
``bash tools/rl/run.sh --simulator maniskill --task-name TwoRobotPickCube-v1 --seed 0``。
初始观测给出实测末端位姿。夹爪机械臂使用 ``move_to`` 和 ``set_gripper``，双臂任务指定左右臂，画笔只使用 ``move_to``。一个目标可以推进多个仿真步。步行和独立手指控制需要另外定义动作。
视觉工具使用任务实际提供的策略相机和观测字段，SAM3 为可选服务。轨迹保存原生累计回报，
有成功判据时记录其值，缺少判据时保存为 JSON ``null``。

在仿真服务器运行 ``RPENT_RUN_MANISKILL_INTEGRATION=1 python -m pytest -v tests/e2e_tests/maniskill/test_environment_smoke.py``
可逐个验收所有注册任务。每项测试会重置环境、读取可用的策略相机，并通过 RPent 执行末端移动。不具备对应动作映射的机器人单列为跳过。

提供训练工具服务
----------------

用同一仿真环境和资产启动 RPC。后端管理分配给它的 GPU 槽和子进程。下面的 LIBERO 服务可同时运行四个环境。

.. code-block:: bash

   python -m robots.rl.remote_manager --simulator libero --gpu-slots 0:4 \
     --sam3-endpoint http://127.0.0.1:28424 --port 28561 \
     --output-dir "$PWD/runs/libero-toolkit"

每种仿真器使用自己的 endpoint。服务默认监听回环地址，跨主机时使用受限转发。资产留在仿真端，观察和工具回执通过 RPC 返回。Planner 和后端使用相同 RPent 版本。训练启动见 `RLinf 指南 <https://github.com/zhexuanxu/RLinf/blob/rpent-ppo/docs/source-zh/rst_source/examples/agentic/rpent_episode_sync.rst>`_。

坐标与动作
----------

模型接口用厘米表示位置、长度和线速度。姿态为固定轴 XYZ 的 roll、pitch、yaw，单位为弧度。单臂取实测安装座为原点，双臂取两个安装座的中点。X 向前，Y 向机器人左侧，Z 向上。感知工具在该坐标系内回报可见表面的几何，原始深度和仿真点云仍用米。

每批最多三个顺序动作，每项控制一个部件。RoboCasa 的手臂目标相对于该项开始时实测的坐标系，包括前一项刚移动底盘或升降的情况。观察和感知回执记录拍摄及结果所用的坐标版本，动作输入只写目标。

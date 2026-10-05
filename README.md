# PiPER MuJoCo Lab

PiPER H 网格 SDF 抓取、批量回放与性能实验。实验代码位于
[contrib/piper_h](contrib/piper_h/README.md)，保留完整机械臂和仅夹爪模式、物体上传、网页参数与回放入口。

官方引擎来自 [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)，以
`external/mujoco_warp` Git submodule 固定在 `cc97eea203bb6d18f5795cd5922a65da1e1051ce`。
本项目的稠密 SDF 和确定性接触改动保存为有序补丁。官方子模块保持原样，运行副本生成到 `.build/mujoco_warp`。
上游文档、基准和 notebooks 可在官方子模块查看；运行时使用带补丁的副本。

## 初始化

需要 uv、Git 和兼容的 NVIDIA CUDA GPU。首次检出或更新补丁后，在本仓库根目录执行：

```bash
git submodule update --init --recursive
uv run --no-project python tools/prepare_mujoco_warp.py
uv sync --locked --extra dev
uv run pre-commit install
uv run python -c 'import mujoco_warp; print(mujoco_warp.__file__)'
```

导入位置应在 `.build/mujoco_warp/mujoco_warp/`。准备脚本不需要已安装的项目依赖，
因此首次运行必须使用 `--no-project`，不能先执行依赖尚未生成路径的 `uv sync`。
不要在官方子模块运行测试或安装；这会创建缓存，使官方 checkout 不再干净。

## 运行与检查

```bash
uv run python contrib/piper_h/grasp_dashboard.py
uv run python contrib/piper_h/grasp.py --headless --engine=warp
uv run python tools/check_engine.py
uv run pre-commit run --all-files
uv run pytest -n 8
```

网页默认 `http://127.0.0.1:8765/`，端口、令牌、历史目录和旧回放格式沿用原行为。
完整模型、接触参数、测试和基准用法见 [PiPER 使用说明](contrib/piper_h/README.md)。
根 pytest 配置同时收集生成副本中的引擎测试、PiPER 测试和准备脚本测试；GPU 验收需要真实 CUDA 主机。
CI 运行可在无 GPU 环境验证的工具和接口测试，不能代替本地完整验收。

## 开发与来源

原外层提交为 `92f301e`；本次拆分还包含改造前未提交的接触排序、约束排序、Hessian 和 PiPER 默认启用改动。
源码清单、补丁来源与维护方式见 [补丁说明](patches/mujoco_warp/README.md) 和
[开发流程](FORK_WORKFLOW.md)。实验结果、虚拟环境及生成源码不进入 Git。
上游 Apache-2.0 许可证及模型的独立许可均保留。

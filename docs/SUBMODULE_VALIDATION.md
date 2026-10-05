# 本地 Submodule 改造验证

日期：2026-10-05。改造分支：`codex/submodule-layout`。
官方子模块：`external/mujoco_warp`，URL 为 `https://github.com/google-deepmind/mujoco_warp.git`。
基线：`cc97eea203bb6d18f5795cd5922a65da1e1051ce`；原外层 HEAD：`92f301e`。
首个改造提交：`0ac8352f96487f3321b76aef957e58667a5c83d1`。

## 已验证内容

- 官方基线加两层补丁后，改造前全部引擎文件的 SHA-256 一致，包括 8 个修改文件和 4 个新增文件；检查清单保存在 `patches/mujoco_warp/source-snapshot.json`。
- 根 uv 项目加载 `.build/mujoco_warp/mujoco_warp`，所有第三方锁定版本与原 `uv.lock` 一致。
- 工具和源码一致性测试 12 项通过，覆盖重复生成、缺失子模块、错误 SHA、脏子模块、补丁冲突、手工增删改、补丁更新和 Git hook 环境隔离。
- 八叉树和稠密 SDF 各运行 16 个环境、20 个物理步；改造前后状态与溢出标记逐位一致。这是结构改造回归验证，不扩大为任意轨迹确定性保证。
- 从本地提交全新 clone，初始化官方 submodule，新建独立 `.venv`，生成源码、锁定安装、安装 hooks、运行 12 项工具与源码检查均通过。
- 独立网页服务使用临时端口及仓库外数据目录，验证页面、参数接口、1 秒 GPU 运行、保存与回放；回放 JPEG 为 640×360，已查看图像确认场景正常。未重启已有服务。
- pre-commit 的 ruff、格式、YAML、uv-lock、引擎和外层 kernel-analyzer 检查通过。

## 测试运行与修正

改造前针对性测试为 133 passed、1 failed。唯一失败是上传预览测试仍期待 640×480，
实际渲染器和文档已经使用 16:9；只将断言更新为 640×360，保留渲染行为。

首次直接并发执行 `MUJOCO_GL=egl uv run pytest -n 8` 为 1631 passed、32 skipped、11 failed，
日志显示显存耗尽及其后的 CUDA 图创建、内核加载、扫描失败。当时显卡还有约 14.5 GiB 的既有进程占用，
未停止任何这些进程。新增 xdist shared CUDA 分组，确保引擎和 PiPER 用例由同一 worker 运行，工具用例仍可并行。
分组使用 tryfirst collection hook，使标记先于 xdist 任务标识生成；实际两 worker 检查的 7 项用例均在同一 worker 通过。
未减少用例、修改物理参数或放宽数值容差。

最终完整测试 `MUJOCO_GL=egl uv run pytest -n 8`：**1644 passed、32 skipped、0 failed**，耗时 207.34 秒。
确定性接触、约束排序、Hessian、稠密 SDF、上传、网页和夹爪测试均已执行；原有上游跳过项保留。

## 本地证据与边界

原提交历史 bundle、源码快照、未提交补丁、哈希清单、被移走目录和仿真输出保存在：
`/home/yangyunchen/Documents/Simulation/mujoco_warp-backup-20261005-155003/`。
该目录还保存干净检出、网页运行记录、回放 JPEG 和完整测试／检查日志，作为本机证据，不加入 Git。

`.build`、虚拟环境、缓存、实验结果不被跟踪；官方子模块保持干净。
Physim 原有两处未提交改动保持原状；本阶段没有迁移、推送或创建 PR。

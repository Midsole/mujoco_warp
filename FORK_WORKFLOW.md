# 官方基线与自定义补丁开发流程

外层仓库管理 PiPER 实验和补丁；`external/mujoco_warp` 只保存官方基线。
运行引擎由 `tools/prepare_mujoco_warp.py` 生成到 `.build/mujoco_warp`，通过根项目的 uv editable 依赖加载。
原外层 Git 历史保留，本次不升级官方基线，也不向官方仓库推送。

## 修改实验代码

直接编辑 `contrib/piper_h`，使用 `uv run` 运行入口或单项测试。官方工具和测试在生成副本内运行。
所有 Python 命令使用 `uv run`。创建 PR 前必须通过 `uv run pytest -n 8`。

## 修改引擎

不要直接修改官方子模块。可在仓库外创建一个临时 Git 仓库：从官方固定 SHA 导出源码，
初始化临时仓库并提交基线，按 `patches/mujoco_warp/series` 顺序应用现有补丁，再提交当前补丁后状态。
在该临时仓库编辑引擎并运行有针对性的检查，导出新增差异为下一个补丁，包括新文件：

```bash
# 在临时工作仓库内；只暂存本轮引擎源码及测试。
git add mujoco_warp
git diff --cached --binary > /absolute/path/to/outer/patches/mujoco_warp/0003-feature.patch
```

在外层仓库将新补丁名追加到 `series`，随后执行：

```bash
uv run --no-project python tools/prepare_mujoco_warp.py
uv sync --locked --extra dev
uv run python tools/check_engine.py
uv run pytest -n 8
```

准备脚本验证官方 HEAD、干净状态和补丁顺序。输入未变时不重建；生成副本有源码修改时拒绝覆盖。
若在 `.build` 中实验过修改，必须先在仓库外保存差异并转成补丁，再将修改恢复为生成清单中的原内容，
或把整个修改副本移至仓库外后重新生成。不要用删除命令丢弃尚未导出的改动。
准备失败不替换上一份运行副本；恢复输入并重新准备后再安装。

## 升级官方基线

这是独立维护任务，不自动跟随 `main`。先备份运行状态、保存回归输出，再获取官方更新并选择明确 SHA。
更新子模块 gitlink 和 `patches/mujoco_warp/base`，重放或重写补丁，重新生成、安装和验证。
核对所有第三方锁定版本变化；补丁冲突时停止，不跳过补丁或降级功能。
提交官方基线变更、补丁变更和必要依赖变化，使新的检出能复现相同引擎。

## 提交与推送

分支默认使用 `codex/` 前缀。提交包括 `.gitmodules`、子模块 gitlink、补丁、外层代码、配置和文档；
不提交 `.build`、缓存或实验结果。外层分支只推送自己的 fork，不向 `google-deepmind/mujoco_warp` 推送。
不添加 AI co-author。PR 描述使用简洁 prose；审阅中的 PR 使用新提交，并逐条回复及解决已处理评论。

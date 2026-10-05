# MuJoCo Warp 自定义补丁

官方来源：`https://github.com/google-deepmind/mujoco_warp.git`。
固定基线：`cc97eea203bb6d18f5795cd5922a65da1e1051ce`，见 `base`。
外层原 HEAD：`92f301e`，分支 `codex/fork-workflow`。

`series` 是唯一应用顺序：

1. `0001-dense-sdf.patch`：官方基线到原 HEAD 的引擎差异，来自提交 `267f2d6` 的稠密 SDF 与梯度缓存。
2. `0002-deterministic-contacts.patch`：原 HEAD 到改造前工作区的引擎差异，包含 8 个修改文件及 4 个新增模块／测试。

PiPER README、rollout 默认开关和新增确定性测试作为外层源文件保留，不放入引擎补丁。
生成副本的 `.source-manifest.json` 保存基线、按顺序排列的补丁 SHA-256、生成文件 SHA-256 和可执行标记。
`source-snapshot.json` 保存改造前引擎文件的 SHA-256，可独立验证拆分没有修改引擎逻辑。

准备使用 Git archive 导出完整官方基线，在临时目录逐个检查并应用补丁，成功后切换运行副本。
官方子模块不安装依赖、不写缓存、不应用补丁。开发与更新流程见根目录 `FORK_WORKFLOW.md`。

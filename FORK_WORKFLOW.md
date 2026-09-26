# Fork 开发与上游基线更新

本仓库的 `origin` 指向个人 fork（`Midsole/mujoco_warp`）。将原仓库配置为 `upstream`，只从它获取更新；自己的分支只推送到 `origin`。

## 首次配置

在本地仓库中执行一次：

```bash
git remote add upstream https://github.com/google-deepmind/mujoco_warp.git
git remote -v
```

确认 `origin` 指向个人 fork，`upstream` 指向 `google-deepmind/mujoco_warp`。如果 `upstream` 已存在，无须重复添加。

## 更新基线

保持 `main` 与原仓库同步，不在 `main` 上提交个人改动。更新前先确认工作区干净：

```bash
git status --short
git fetch upstream
git switch main
git merge --ff-only upstream/main
git push origin main
```

`--ff-only` 在本地 `main` 已有独立提交时会停止，不会自动合并或改写历史。此时先检查差异并处理这些提交，不要直接强制推送：

```bash
git log --oneline --left-right main...upstream/main
```

## 开发自己的功能

从更新后的 `main` 创建分支，在该分支提交修改，并明确推送到个人 fork：

```bash
git switch main
git switch -c my-feature
# 修改文件并提交
git push -u origin my-feature
```

上游继续更新时，先按上节更新 `main`，再将新基线合入长期开发分支：

```bash
git switch my-feature
git merge main
git push origin my-feature
```

如果分支只有自己使用，也可以用 `git rebase main` 代替 `git merge main`，但 rebase 会改写该分支的提交历史。已共享的分支优先使用 merge。

推送时始终明确写 `origin`。配置 `upstream` 本身不会推送任何改动；只有主动向它执行 `git push` 才会尝试写入原仓库。

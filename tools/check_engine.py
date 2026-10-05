"""Check generated Python files touched by the maintained engine patches."""

import subprocess

from prepare_mujoco_warp import ROOT
from prepare_mujoco_warp import prepare


def main():
  engine = prepare()
  paths = set()
  for patch in (ROOT / "patches/mujoco_warp").glob("*.patch"):
    for line in patch.read_text().splitlines():
      if line.startswith("+++ b/") and line.endswith(".py"):
        paths.add(engine / line[6:])
  files = sorted(str(path) for path in paths)
  for args in (["ruff", "check"], ["ruff", "format", "--check"]):
    subprocess.run(["uv", "run", *args, "--config", str(ROOT / "pyproject.toml"), *files], cwd=ROOT, check=True)
  subprocess.run(
    [
      "uv",
      "run",
      "python",
      str(ROOT / "contrib/kernel_analyzer/kernel_analyzer/cli.py"),
      "--types",
      str(engine / "mujoco_warp/_src/types.py"),
      *files,
    ],
    cwd=ROOT,
    check=True,
  )


if __name__ == "__main__":
  main()

"""Generate the patched engine without modifying the official submodule.

Bootstrap with ``uv run --no-project python tools/prepare_mujoco_warp.py``.
"""

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ".source-manifest.json"


def git_environment():
  """Prevent parent Git hooks from redirecting submodule and patch operations."""
  return {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}


def git(directory, *args):
  """Run Git without permitting mutations in the official checkout."""
  return subprocess.check_output(["git", "-C", str(directory), *args], stderr=subprocess.STDOUT, env=git_environment())


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(directory):
  """Ignore only known build and runtime artifacts when checking generated sources."""
  result = {}
  for path in directory.rglob("*"):
    relative = path.relative_to(directory)
    if any(
      part in {"__pycache__", ".pytest_cache", ".ruff_cache", "build", ".git", ".venv"} or part.endswith(".egg-info")
      for part in relative.parts
    ):
      continue
    if path.name == MANIFEST or path.suffix in {".pyc", ".pyo"} or path.name == "MUJOCO_LOG.TXT":
      continue
    if path.is_symlink():
      raise RuntimeError(f"Unexpected symlink in generated source: {relative}")
    if path.is_file():
      result[relative.as_posix()] = {"sha256": digest(path), "executable": bool(path.stat().st_mode & 0o111)}
  return result


def inputs(root):
  """Validate the pinned submodule and read the ordered patch inputs."""
  official = root / "external/mujoco_warp"
  patches = root / "patches/mujoco_warp"
  base = (patches / "base").read_text().strip()
  if len(base) != 40 or any(c not in "0123456789abcdef" for c in base):
    raise RuntimeError("The base must be a full Git SHA")
  if not (official / ".git").exists():
    raise RuntimeError("Submodule missing; run git submodule update --init --recursive")
  if git(official, "rev-parse", "HEAD").decode().strip() != base:
    raise RuntimeError(f"Submodule HEAD must match pinned base {base}")
  if git(official, "status", "--porcelain", "--untracked-files=all", "--ignored").strip():
    raise RuntimeError("Official submodule must be clean, including untracked and ignored files")
  names = [line.strip() for line in (patches / "series").read_text().splitlines() if line.strip() and not line.startswith("#")]
  if not names or len(names) != len(set(names)):
    raise RuntimeError("Patch series must be nonempty and contain no duplicates")
  for name in names:
    if Path(name).name != name or not name.endswith(".patch"):
      raise RuntimeError(f"Invalid patch name: {name}")
  return (
    official,
    [patches / name for name in names],
    {
      "schema": 1,
      "base": base,
      "patches": [{"name": name, "sha256": digest(patches / name)} for name in names],
    },
  )


def prepare(root=ROOT):
  """Prepare transactionally; never overwrite manually edited generated source."""
  root = Path(root).resolve()
  official, patches, signature = inputs(root)
  build = root / ".build"
  target = build / "mujoco_warp"
  if target.is_symlink():
    raise RuntimeError("Generated source directory must not be a symlink")
  if target.exists():
    try:
      previous = json.loads((target / MANIFEST).read_text())
    except (OSError, ValueError) as error:
      raise RuntimeError("Generated source has no valid manifest; preserve it before regenerating") from error
    if source_files(target) != previous["files"]:
      raise RuntimeError("Generated source was manually modified; export edits to a patch before regenerating")
    if previous["inputs"] == signature:
      print(f"Unchanged: {target}")
      return target
  build.mkdir(exist_ok=True)
  staging = Path(tempfile.mkdtemp(prefix=".mujoco-warp-", dir=build))
  old = None
  try:
    archive = git(official, "archive", "--format=tar", signature["base"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
      for member in tar.getmembers():
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
          raise RuntimeError(f"Unsafe archive member: {member.name}")
      options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
      tar.extractall(staging, **options)
    # Run outside the parent's Git discovery scope, without creating another repository.
    env = git_environment()
    env["GIT_CEILING_DIRECTORIES"] = str(build)
    for patch in patches:
      for check in (True, False):
        command = ["git", "apply", "--whitespace=error"]
        if check:
          command.append("--check")
        subprocess.run([*command, str(patch)], cwd=staging, env=env, check=True)
    manifest = {"inputs": signature, "files": source_files(staging)}
    (staging / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if target.exists():
      old = Path(tempfile.mkdtemp(prefix=".previous-mujoco-warp-", dir=build))
      old.rmdir()
      target.rename(old)
    try:
      staging.rename(target)
    except BaseException:
      if old:
        old.rename(target)
      raise
    if old:
      shutil.rmtree(old)
    print(f"Prepared: {target}")
    return target
  finally:
    if staging.exists():
      shutil.rmtree(staging)


if __name__ == "__main__":
  try:
    prepare()
  except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
    raise SystemExit(str(error)) from error

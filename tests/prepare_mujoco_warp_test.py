"""Exercise reproducible generation and preservation of source on failures."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("prepare", Path(__file__).resolve().parents[1] / "tools/prepare_mujoco_warp.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


@pytest.fixture
def project(tmp_path):
  official = tmp_path / "external/mujoco_warp"
  official.mkdir(parents=True)
  subprocess.run(["git", "init", "-q", str(official)], check=True)
  (official / "sample.txt").write_text("before\n")
  prepare.git(official, "add", "sample.txt")
  prepare.git(official, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")
  patches = tmp_path / "patches/mujoco_warp"
  patches.mkdir(parents=True)
  (patches / "base").write_bytes(prepare.git(official, "rev-parse", "HEAD"))
  (patches / "series").write_text("sample.patch\n")
  (patches / "sample.patch").write_text(
    "diff --git a/sample.txt b/sample.txt\n--- a/sample.txt\n+++ b/sample.txt\n@@ -1 +1 @@\n-before\n+after\n"
  )
  return tmp_path


def test_repeat_is_noop_and_official_stays_clean(project):
  target = prepare.prepare(project)
  assert (target / "sample.txt").read_text() == "after\n"
  before = (target / prepare.MANIFEST).stat().st_mtime_ns
  assert prepare.prepare(project) == target
  assert (target / prepare.MANIFEST).stat().st_mtime_ns == before
  assert not prepare.git(project / "external/mujoco_warp", "status", "--porcelain")


@pytest.mark.parametrize("problem", ["missing", "wrong_sha", "dirty", "untracked"])
def test_invalid_submodule_preserves_previous_output(project, problem):
  target = prepare.prepare(project)
  official = project / "external/mujoco_warp"
  if problem == "missing":
    (official / ".git").rename(official / "saved-git")
  elif problem == "wrong_sha":
    (project / "patches/mujoco_warp/base").write_text("0" * 40)
  elif problem == "dirty":
    (official / "sample.txt").write_text("edited\n")
  else:
    (official / "unexpected.txt").write_text("untracked\n")
  with pytest.raises(RuntimeError):
    prepare.prepare(project)
  assert (target / "sample.txt").read_text() == "after\n"


@pytest.mark.parametrize("edit", ["change", "delete", "add"])
def test_manual_edits_are_never_overwritten(project, edit):
  target = prepare.prepare(project)
  if edit == "change":
    (target / "sample.txt").write_text("manual\n")
  elif edit == "delete":
    (target / "sample.txt").unlink()
  else:
    (target / "new.txt").write_text("manual\n")
  snapshot = prepare.source_files(target)
  with pytest.raises(RuntimeError, match="manually modified"):
    prepare.prepare(project)
  assert prepare.source_files(target) == snapshot


def test_patch_conflict_preserves_previous_output(project):
  target = prepare.prepare(project)
  patch = project / "patches/mujoco_warp/sample.patch"
  patch.write_text(patch.read_text().replace("-before", "-conflict"))
  with pytest.raises(subprocess.CalledProcessError):
    prepare.prepare(project)
  assert (target / "sample.txt").read_text() == "after\n"
  assert not list((project / ".build").glob(".mujoco-warp-*"))


def test_changed_patch_regenerates_and_ignores_python_cache(project):
  target = prepare.prepare(project)
  cache = target / "__pycache__"
  cache.mkdir()
  (cache / "sample.pyc").write_bytes(b"cache")
  patch = project / "patches/mujoco_warp/sample.patch"
  patch.write_text(patch.read_text().replace("+after", "+updated"))
  prepare.prepare(project)
  assert (target / "sample.txt").read_text() == "updated\n"


def test_parent_git_hook_environment_is_isolated(project, monkeypatch):
  monkeypatch.setenv("GIT_INDEX_FILE", str(project / "nonexistent-parent-index"))
  monkeypatch.setenv("GIT_DIR", str(project / "nonexistent-parent-git"))
  target = prepare.prepare(project)
  assert (target / "sample.txt").read_text() == "after\n"

"""Reference pairing, historical compatibility and independent-world cache checks."""

import json
import os
from types import SimpleNamespace

import grasp_config
import gripper_reference as reference
import numpy as np
import pytest


def test_old_history_and_cpu_restriction():
  assert grasp_config.validate_config(grasp_config.history_config({}))["robot_mode"] == "full_arm"
  with pytest.raises(ValueError, match="Warp"):
    grasp_config.validate_config({"robot_mode": "gripper_only", "reference_run_id": "a" * 32, "engine": "c"})


def test_reference_pairing_and_cache(tmp_path, monkeypatch):
  run_id = "a" * 32
  source = tmp_path / run_id
  source.mkdir()
  full = grasp_config.validate_config({"nworld": 16})
  paired = {**full, "robot_mode": "gripper_only", "reference_run_id": run_id}
  (source / "config.json").write_text(json.dumps(full))
  (source / "status.json").write_text(json.dumps({"state": "completed", "passed": True}))
  for world in range(16):
    path = source if world == 0 else source / "worlds" / f"{world + 1:04d}"
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "trace.npz", qpos=np.full((2, 3), world), qvel=np.full((2, 2), -world))
  assert reference.check_reference(tmp_path, run_id, paired) == source
  with pytest.raises(ValueError, match="配对参数"):
    reference.check_reference(tmp_path, run_id, {**paired, "cat_mass": 0.2})
  model = SimpleNamespace(nq=3, nv=2)
  for name in (
    "body_mass",
    "body_inertia",
    "body_pos",
    "body_quat",
    "body_parentid",
    "body_jntadr",
    "jnt_type",
    "jnt_pos",
    "jnt_axis",
    "geom_pos",
    "geom_quat",
    "geom_type",
    "geom_dataid",
    "mesh_vert",
    "mesh_face",
    "actuator_trnid",
  ):
    setattr(model, name, np.zeros(1))
  monkeypatch.setattr(reference.grasp, "build_model", lambda *args, **kwargs: model)
  monkeypatch.setattr(reference.grasp, "make_trajectory", lambda *args: {"ctrl": np.zeros((2, 1))})
  calls = []

  def export(model, q, v):
    calls.append(True)
    np.testing.assert_array_equal(q[:, 0, 0], np.arange(16))
    np.testing.assert_array_equal(v[:, 0, 0], -np.arange(16))
    motion = np.zeros((16, 2, 19), np.float32)
    motion[:, :, 0] = q[:, :, 0]
    return motion

  monkeypatch.setattr(reference, "export_motion", export)
  monkeypatch.setattr(reference, "reduced_trajectory", lambda a, b, c, motion: {"root_motion": motion})
  for i in range(2):
    destination = tmp_path / f"paired{i}"
    destination.mkdir()
    trajectory, metrics = reference.prepare(tmp_path, run_id, paired, model, destination)
    assert metrics["reference_cache_hit"] == bool(i)
    np.testing.assert_array_equal(trajectory["root_motion"][:, 0, 0], np.arange(16))
    metadata = json.loads((destination / "reference.json").read_text())
    assert metadata["reference_run_id"] == run_id and len(metadata["model_sha256"]) == 64
  assert len(calls) == 1
  # An interrupted metadata write must be rebuilt, rather than reused as a valid cache.
  cache = next((tmp_path / "reference_cache").iterdir())
  (cache / "metadata.json").write_text("{")
  destination = tmp_path / "repaired"
  destination.mkdir()
  _, metrics = reference.prepare(tmp_path, run_id, paired, model, destination)
  assert not metrics["reference_cache_hit"] and len(calls) == 2
  # Content changes invalidate the cache even after restoring size and timestamps.
  path = source / "trace.npz"
  stat = path.stat()
  np.savez(path, qpos=np.zeros((2, 3), dtype=int), qvel=np.ones((2, 2), dtype=int))
  assert path.stat().st_size == stat.st_size
  os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
  expected_velocities = -np.arange(16)
  expected_velocities[0] = 1

  def changed_export(model, q, v):
    calls.append(True)
    np.testing.assert_array_equal(v[:, 0, 0], expected_velocities)
    return np.zeros((16, 2, 19), np.float32)

  monkeypatch.setattr(reference, "export_motion", changed_export)
  destination = tmp_path / "changed"
  destination.mkdir()
  _, metrics = reference.prepare(tmp_path, run_id, paired, model, destination)
  assert not metrics["reference_cache_hit"] and len(calls) == 3
  (source / "worlds/0016/trace.npz").unlink()
  with pytest.raises(ValueError, match="逐场景状态"):
    reference.check_reference(tmp_path, run_id, paired)

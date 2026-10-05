"""Export and cache immutable, per-world end-effector reference trajectories."""

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path

import grasp
import numpy as np
from gripper_only import export_motion
from gripper_only import reduced_trajectory

EXPORT_VERSION = 3


def file_sha256(path):
  """Hash source contents so preserving file size and timestamps cannot hide changes."""
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def check_reference(root, run_id, config):
  """Validate the pairing before the worker starts; ignore only the two mode selectors."""
  from grasp_config import history_config
  from grasp_config import validate_config

  if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
    raise ValueError("参考运行编号无效")
  directory = Path(root) / run_id
  try:
    status = json.loads((directory / "status.json").read_text())
    reference = validate_config(history_config(json.loads((directory / "config.json").read_text())))
  except FileNotFoundError as exc:
    raise ValueError("参考运行不存在") from exc
  if status["state"] != "completed" or not status.get("passed") or reference["robot_mode"] != "full_arm":
    raise ValueError("参考运行必须是全部场景验收通过的完整机械臂运行")
  if reference["engine"] != "warp":
    raise ValueError("请选择 Warp 完整机械臂参考运行")
  differences = [key for key in reference if key not in ("robot_mode", "reference_run_id") and config[key] != reference[key]]
  if differences:
    raise ValueError("配对参数必须与参考运行一致：" + ", ".join(differences))
  for world in range(1, config["nworld"] + 1):
    path = directory if world == 1 else directory / "worlds" / f"{world:04d}"
    if not (path / "trace.npz").is_file():
      raise ValueError("参考运行缺少逐场景状态，请重新运行完整机械臂")
  return directory


def prepare(root, run_id, config, model, destination):
  """Return mapped controls and root motion; preparation is not charged to physics steps."""
  started = time.perf_counter()
  source = check_reference(root, run_id, config)
  full_config = {**config, "robot_mode": "full_arm", "reference_run_id": ""}
  mesh = {"object_path": source / "object.obj"} if config["object_shape"] == "uploaded" else {}
  full = grasp.build_model(full_config, **mesh)
  trajectory = grasp.make_trajectory(full, full_config)
  count = len(trajectory["ctrl"])
  paths = [
    (source if world == 1 else source / "worlds" / f"{world:04d}") / "trace.npz" for world in range(1, config["nworld"] + 1)
  ]
  # The config and arrays include the exported model's geometry, joints and inertias.
  model_hash = hashlib.sha256()
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
    model_hash.update(getattr(full, name).tobytes())
  signature = {
    "export_version": EXPORT_VERSION,
    "reference_run_id": run_id,
    "config": full_config,
    "model_sha256": model_hash.hexdigest(),
    "sources": [(str(p.relative_to(source)), p.stat().st_size, p.stat().st_mtime_ns) for p in paths],
    "source_sha256": [file_sha256(p) for p in paths],
  }
  key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
  cache = Path(root) / "reference_cache" / key
  cache.mkdir(parents=True, exist_ok=True)
  cached = False
  try:
    metadata = json.loads((cache / "metadata.json").read_text())
    cached_motion = np.load(cache / "motion.npy", mmap_mode="r", allow_pickle=False)
    cached = metadata.get("cache_key") == key and cached_motion.shape == (config["nworld"], count, 19)
    cached = cached and cached_motion.dtype == np.float32
    del cached_motion
  except (OSError, ValueError):
    pass
  if not cached:
    positions = np.empty((config["nworld"], count, full.nq), np.float32)
    velocities = np.empty((config["nworld"], count, full.nv), np.float32)
    state_hash = hashlib.sha256()
    for world, path in enumerate(paths):
      with np.load(path) as archive:
        if not {"qpos", "qvel"}.issubset(archive.files):
          raise ValueError("参考运行缺少速度状态，请重新运行完整机械臂")
        q, v = archive["qpos"], archive["qvel"]
        if q.shape != positions[world].shape or v.shape != velocities[world].shape:
          raise ValueError("参考轨迹形状与模型或任务长度不一致，请重新运行完整机械臂")
        positions[world], velocities[world] = q, v
        state_hash.update(q.tobytes())
        state_hash.update(v.tobytes())
    motion = export_motion(full, positions, velocities)
    del positions, velocities
    if not np.isfinite(motion).all():
      raise ValueError("参考末端轨迹包含非有限值")
    np.save(cache / "motion.part.npy", motion)
    os.replace(cache / "motion.part.npy", cache / "motion.npy")
    metadata = {
      **signature,
      "cache_key": key,
      "states_sha256": state_hash.hexdigest(),
      "shape": list(motion.shape),
      "coordinates": "world; quaternion wxyz; velocities at root origin; pre-step poses",
      "acceleration": "forward difference of analytic body-origin linear/angular velocities",
    }
    (cache / "metadata.part.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    os.replace(cache / "metadata.part.json", cache / "metadata.json")
  shutil.copyfile(cache / "motion.npy", Path(destination) / "root_motion.npy")
  shutil.copyfile(cache / "metadata.json", Path(destination) / "reference.json")
  motion = np.load(Path(destination) / "root_motion.npy", mmap_mode="r")
  mapped = reduced_trajectory(full, model, trajectory, motion)
  return mapped, {"reference_prepare_seconds": time.perf_counter() - started, "reference_cache_hit": cached}


def main():
  """Export a dashboard reference independently of a new simulation."""
  import argparse

  from grasp_config import history_config
  from grasp_config import validate_config

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--data-dir", type=Path, default=Path.home() / ".local/share/mujoco_warp/piper_h_runs")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if not re.fullmatch(r"[0-9a-f]{32}", args.run_id):
    parser.error("参考运行编号无效")
  source = args.data_dir / args.run_id
  config = history_config(json.loads((source / "config.json").read_text()))
  config.update(robot_mode="gripper_only", reference_run_id=args.run_id)
  config = validate_config(config)
  check_reference(args.data_dir, args.run_id, config)
  mesh = {"object_path": source / "object.obj"} if config["object_shape"] == "uploaded" else {}
  model = grasp.build_model(config, **mesh)
  args.output.mkdir(parents=True, exist_ok=True)
  _, metrics = prepare(args.data_dir, args.run_id, config, model, args.output)
  print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()

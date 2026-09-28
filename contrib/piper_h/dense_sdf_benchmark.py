"""Compare dense cached distance/gradient queries with their source octrees."""

import argparse
import json
from pathlib import Path

import finger_cost_profile as profiler
import mujoco
import numpy as np
import warp as wp
from collision_cost_profile import STAGES
from collision_cost_profile import average
from dense_sdf import attach_dense_sdf
from dense_sdf import reference_volume
from finger_collision_benchmark import collision_measure
from grasp import build_model
from grasp import make_trajectory
from grasp import validate
from grasp_config import validate_config
from grasp_warp import rollout_warp
from parallel_efficiency_benchmark import full_task_measure

import mujoco_warp as mjw
from mujoco_warp._src.collision_sdf import DenseSDF
from mujoco_warp._src.collision_sdf import VolumeData
from mujoco_warp._src.collision_sdf import attach_dense
from mujoco_warp._src.collision_sdf import sample_volume_grad
from mujoco_warp._src.collision_sdf import sample_volume_sdf
from mujoco_warp._src.types import BroadphaseType


@wp.kernel
def query(volume: VolumeData, grids: DenseSDF, mesh: int, points: wp.array[wp.vec3], output_out: wp.array[wp.vec4]):
  i = wp.tid()
  volume = attach_dense(volume, grids, mesh)
  p = points[i]
  distance = sample_volume_sdf(p, volume)
  gradient = sample_volume_grad(p, volume)
  output_out[i] = wp.vec4(distance, gradient[0], gradient[1], gradient[2])


def query_measure(volume, grids, mesh, points):
  output = wp.empty(len(points), dtype=wp.vec4)
  with wp.ScopedCapture() as capture:
    wp.launch(query, dim=len(points), inputs=[volume, grids, mesh, points, output])
  for _ in range(10):
    wp.capture_launch(capture.graph)
  wp.synchronize()
  timings = []
  for _ in range(5):
    start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    wp.record_event(start)
    for _ in range(100):
      wp.capture_launch(capture.graph)
    wp.record_event(end)
    wp.synchronize_event(end)
    timings.append(wp.get_event_elapsed_time(start, end) / 100)
  return output.numpy(), timings


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--resolutions", nargs="+", type=int, default=[257])
  parser.add_argument("--gradient-mode", choices=("vertex", "cell"), default="cell")
  parser.add_argument("--worlds", nargs="*", type=int, default=[])
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--validate", action="store_true", help="Record and validate the complete single-world grasp")
  parser.add_argument("--profile", action="store_true", help="Sample single-world broadphase/narrowphase CUDA times")
  parser.add_argument("--collision-snapshots", type=Path, help="Compare frozen collision states from a snapshots.npz file")
  parser.add_argument("--output", type=Path, default=Path("contrib/piper_h/results/dense_sdf/benchmark.json"))
  args = parser.parse_args()
  if args.repeats < 1 or any(n < 1 for n in args.worlds) or any(not 3 <= n <= 513 for n in args.resolutions):
    parser.error("repeats/worlds must be positive and resolutions must be in [3, 513]")
  if not wp.is_cuda_available():
    parser.error("CUDA is required")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  model = build_model()
  warp_model = mjw.put_model(model)
  if args.profile and warp_model.opt.broadphase != BroadphaseType.NXN:
    parser.error("collision stage profiling requires NXN broadphase")
  trajectory = make_trajectory(model) if args.worlds or args.validate or args.profile else None
  result = {
    "device": wp.get_device().name,
    "mujoco_version": mujoco.__version__,
    "warp_version": wp.__version__,
    "query_points_per_mesh": 65536,
    "config": validate_config({}),
    "resolutions": {},
    "full_task": [],
  }
  rng = np.random.default_rng(42)
  references = {}
  # Include exterior points to exercise the existing box extension and its gradient.
  for mesh in sorted(set(model.geom_dataid[model.geom_type == mujoco.mjtGeom.mjGEOM_SDF])):
    mesh = int(mesh)
    volume = reference_volume(model, warp_model, mesh)
    points = wp.array(
      np.asarray(volume.center) + rng.uniform(-1.1, 1.1, (65536, 3)) * np.asarray(volume.half_size), dtype=wp.vec3
    )
    values, times = query_measure(volume, DenseSDF(), mesh, points)
    references[mesh] = (volume, points, values, times)
  for resolution in args.resolutions:
    metadata = attach_dense_sdf(model, warp_model, resolution, gradient_mode=args.gradient_mode)
    rows = []
    for mesh, (volume, points, reference, times) in references.items():
      values, dense_times = query_measure(volume, warp_model.dense_sdf, mesh, points)
      error = np.abs(values[:, 0] - reference[:, 0])
      a, b = values[:, 1:].astype(np.float64), reference[:, 1:].astype(np.float64)
      norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
      valid = norms > 1e-8
      angle = np.degrees(np.arccos(np.clip(np.sum(a[valid] * b[valid], axis=1) / norms[valid], -1, 1)))
      near = np.abs(reference[:, 0]) < 0.001
      near_angle = angle[near[valid]]
      worst = int(np.flatnonzero(valid)[np.argmax(angle)])
      rows.append(
        {
          "mesh": model.mesh(mesh).name,
          "finite": bool(np.isfinite(values).all()),
          "distance_error_m_p50_p95_p99_max": np.percentile(error, [50, 95, 99, 100]).tolist(),
          "gradient_angle_deg_p50_p95_p99_max": np.percentile(angle, [50, 95, 99, 100]).tolist(),
          "gradient_vector_error_p50_p95_p99_max": np.percentile(np.linalg.norm(a - b, axis=1), [50, 95, 99, 100]).tolist(),
          "near_surface_points": int(near.sum()),
          "near_surface_distance_error_m_p50_p95_p99_max": np.percentile(error[near], [50, 95, 99, 100]).tolist(),
          "near_surface_gradient_angle_deg_p50_p95_p99_max": np.percentile(near_angle, [50, 95, 99, 100]).tolist(),
          "zero_gradient_points": int((~valid).sum()),
          "worst_angle_sample": {
            "point": points.numpy()[worst].tolist(),
            "reference": reference[worst].tolist(),
            "dense": values[worst].tolist(),
          },
          "octree_ms": times,
          "dense_ms": dense_times,
          "query_speedup": float(np.median(times) / np.median(dense_times)),
        }
      )
    result["resolutions"][str(resolution)] = {**metadata, "queries": rows}
    if args.validate:
      trace = rollout_warp(model, trajectory, warp_model=warp_model)
      result["resolutions"][str(resolution)]["grasp_validation"] = validate(model, trajectory, trace)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["resolutions"][str(resolution)]), flush=True)
    if args.collision_snapshots:
      frozen = []
      grids = warp_model.dense_sdf
      with np.load(args.collision_snapshots) as snapshots:
        for phase in ("grip", "carry"):
          for nworld in (1, 512):
            for mode in ("octree", "dense"):
              warp_model.dense_sdf = grids if mode == "dense" else DenseSDF()
              measured = collision_measure(
                model,
                warp_model,
                snapshots[f"{phase}_qpos"],
                snapshots[f"{phase}_qvel"],
                nworld,
                validate_config({}),
                args.repeats,
              )
              frozen.append({"mode": mode, "phase": phase, "nworld": nworld, **measured})
      warp_model.dense_sdf = grids
      result["resolutions"][str(resolution)]["frozen_collision"] = frozen
      args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.profile:
      runs = []
      grids = warp_model.dense_sdf
      previous_stages = profiler.STAGES
      try:
        profiler.STAGES = STAGES
        for repeat in range(args.repeats):
          for mode in ["octree", "dense"] if repeat % 2 == 0 else ["dense", "octree"]:
            warp_model.dense_sdf = grids if mode == "dense" else DenseSDF()
            run = profiler.measure(model, warp_model, trajectory, True, 20, validate_config({}))
            runs.append({"mode": mode, "repeat": repeat, "average_ms": average(run["samples"]), **run})
            result["resolutions"][str(resolution)]["profile"] = runs
            args.output.write_text(json.dumps(result, indent=2) + "\n")
      finally:
        profiler.STAGES = previous_stages
        warp_model.dense_sdf = grids
    for nworld in args.worlds:
      for repeat in range(args.repeats):
        grids = warp_model.dense_sdf
        for mode in ["octree", "dense"] if repeat % 2 == 0 else ["dense", "octree"]:
          warp_model.dense_sdf = grids if mode == "dense" else DenseSDF()
          measured = full_task_measure(model, warp_model, trajectory, nworld, 256, 1024)
          result["full_task"].append({"mode": mode, "resolution": resolution, "repeat": repeat, **measured})
          args.output.write_text(json.dumps(result, indent=2) + "\n")
        warp_model.dense_sdf = grids


if __name__ == "__main__":
  main()

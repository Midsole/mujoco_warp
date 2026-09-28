"""Compare box and SDF fingers over identical complete Warp control trajectories."""

import argparse
import json
import statistics
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp
from grasp import build_model
from grasp import make_trajectory
from grasp import validate
from grasp_config import validate_config
from grasp_warp import rollout_warp
from parallel_efficiency_benchmark import full_task_measure

import mujoco_warp as mjw
from mujoco_warp._src.types import OverflowType

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "cube_sdf" / "comparison.json"


def collision_overflow(data):
  """Collision alone does not run step's capacity checks in _next_time."""
  overflow = int(np.bitwise_or.reduce(data.overflow.numpy()))
  if int(data.nacon.numpy()[0]) > data.naconmax:
    overflow |= int(OverflowType.NARROWPHASE)
  if int(data.ncollision.numpy()[0]) > data.naconmax:
    overflow |= int(OverflowType.BROADPHASE)
  return overflow


def collision_measure(model, warp_model, qpos, qvel, nworld, config, repeats, steps=500):
  """Time collision detection at an identical frozen state, independent of grasp success."""
  if nworld < 1 or repeats < 1 or steps < 1:
    raise ValueError("nworld, repeats and steps must be positive")
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.qvel[:] = qvel
  mujoco.mj_forward(model, data)
  d = mjw.put_data(model, data, nworld=nworld, nconmax=config["nconmax"], njmax=config["njmax"])
  with wp.ScopedCapture() as capture:
    mjw.collision(warp_model, d)
  for _ in range(10):
    wp.capture_launch(capture.graph)
  wp.synchronize()
  elapsed = []
  for _ in range(repeats):
    started = time.perf_counter()
    for _ in range(steps):
      wp.capture_launch(capture.graph)
    wp.synchronize()
    elapsed.append(time.perf_counter() - started)
  return {
    "seconds": statistics.median(elapsed),
    "repeats_seconds": elapsed,
    "steps": steps,
    "contacts_per_world": float(d.nacon.numpy()[0] / nworld),
    "overflow": collision_overflow(d),
  }


def write_report(path, output):
  """Write raw measurements and a compact comparison using medians across repeats."""
  path.write_text(json.dumps(output, indent=2) + "\n")
  lines = [
    "# PiPER finger collision comparison",
    "",
    f"Device: {output['device']}; MuJoCo {output['mujoco_version']}; Warp {output['warp_version']}.",
    f"Object: {output['config']['object_shape']}; native mesh octree SDF, depth {output['config']['sdf_depth']}; no analytic plugin.",
    f"Each run: {output['steps_per_world']} steps at {output['config']['timestep']} s; "
    f"{output['repeats']} repetitions per configuration.",
    "",
    f"Environment note: {output.get('environment_note') or 'No isolation information recorded.'}",
    "",
    "The measured interval includes control updates, physics steps and host graph launches, with a final GPU synchronization.",
    "Model compilation, model upload, CUDA graph preparation, warmup, replay recording and result download are excluded.",
    "Both modes use the same controls, mass, inertia, friction, solver, SDF depth and contact capacities.",
    "Box/SDF order alternates between repetitions. These are complete trajectories, so contact histories may differ.",
    "The placement metric only checks final XY error <= 5 cm; it is not the complete grasp acceptance test.",
    "",
    "| Worlds | Box seconds | SDF seconds | Box world-steps/s | SDF world-steps/s | SDF / box time |",
    "| ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for nworld in output["worlds"]:
    timings = {}
    for mode in ("box", "sdf"):
      runs = [run for run in output["runs"] if run["nworld"] == nworld and run["finger_collision"] == mode]
      if runs:
        timings[mode] = statistics.median(run["seconds"] for run in runs)
    if len(timings) != 2:
      continue
    box, sdf = timings["box"], timings["sdf"]
    steps = nworld * output["steps_per_world"]
    lines.append(f"| {nworld} | {box:.3f} | {sdf:.3f} | {steps / box:,.0f} | {steps / sdf:,.0f} | {sdf / box:.2f}x |")
  lines.extend(["", "Compilation times (seconds; separate from simulation timing):", ""])
  for mode, seconds in output["compile_seconds"].items():
    lines.append(f"- {mode}: {seconds:.3f}")
  lines.extend(
    [
      "",
      "## Frozen-state collision detection",
      "",
      "Both modes use exactly the same saved box-rollout qpos/qvel, with no dynamics between collision queries.",
      "Timings include graph launch overhead; geometric differences can change contact counts even at the same pose.",
      "",
      "| Worlds | State | Box ms/query | SDF ms/query | SDF / box time | Box / SDF contacts per world |",
      "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
  )
  for nworld in output["worlds"]:
    for phase in ("grip", "carry"):
      results = {r["finger_collision"]: r for r in output["collision_runs"] if r["nworld"] == nworld and r["phase"] == phase}
      if len(results) != 2:
        continue
      box, sdf = results["box"], results["sdf"]
      lines.append(
        f"| {nworld} | {phase} | {1000 * box['seconds'] / box['steps']:.3f} | "
        f"{1000 * sdf['seconds'] / sdf['steps']:.3f} | {sdf['seconds'] / box['seconds']:.2f}x | "
        f"{box['contacts_per_world']:.1f} / {sdf['contacts_per_world']:.1f} |"
      )
  if output.get("diagnostic_metrics"):
    lines.extend(
      [
        "",
        "## Separate single-world grasp diagnostics",
        "",
        "These recorded runs are outside the timed measurements. Original acceptance thresholds are retained.",
        "",
        "| Fingers | Carried height (m) | Both-finger contact fraction | Final XY error (m) | Full acceptance |",
        "| --- | ---: | ---: | ---: | --- |",
      ]
    )
    for mode, metrics in output["diagnostic_metrics"].items():
      lines.append(
        f"| {mode} | {metrics['minimum_carried_height_m']:.4f} | {metrics['carried_contact_fraction']:.3f} | "
        f"{metrics['placement_error_m']:.4f} | {metrics['passed']} |"
      )
  path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--worlds", type=int, nargs="+", default=[1, 16, 64, 256])
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--sdf-depth", type=int, default=8)
  parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  parser.add_argument("--object-shape", choices=("cube", "cat"), default="cube")
  parser.add_argument("--note", default="", help="Record shared GPU load or other timing conditions in the report")
  args = parser.parse_args()
  if args.repeats < 1 or any(nworld < 1 for nworld in args.worlds):
    parser.error("world counts and repeats must be positive")
  if not wp.is_cuda_available():
    raise RuntimeError("CUDA device required")
  wp.config.log_level = wp.LOG_WARNING
  config = validate_config({"sdf_depth": args.sdf_depth, "object_shape": args.object_shape})
  args.output.parent.mkdir(parents=True, exist_ok=True)
  output = {
    "device": wp.get_device("cuda:0").name,
    "mujoco_version": mujoco.__version__,
    "warp_version": wp.__version__,
    "config": config,
    "environment_note": args.note,
    "collision_meshes": {},
    "worlds": args.worlds,
    "repeats": args.repeats,
    "compile_seconds": {},
    "runs": [],
    "collision_runs": [],
  }
  models = {}
  for mode in ("box", "sdf"):
    started = time.perf_counter()
    models[mode] = build_model({**config, "finger_collision": mode})
    output["compile_seconds"][mode] = time.perf_counter() - started
    model = models[mode]
    output["collision_meshes"][mode] = [
      {
        "name": model.mesh(mesh_id).name,
        "triangles": int(model.mesh_facenum[mesh_id]),
        "octree_nodes": int(model.mesh_octnum[mesh_id]),
        "max_depth": int(model.oct_depth[start : start + count].max()),
      }
      for mesh_id, (start, count) in enumerate(zip(model.mesh_octadr, model.mesh_octnum, strict=True))
      if count > 0
    ]
    print(f"{mode}: model compiled in {output['compile_seconds'][mode]:.2f}s", flush=True)
  trajectory = make_trajectory(models["box"])
  output["steps_per_world"] = len(trajectory["ctrl"])
  # Save a reproducible reference for the isolated collision comparison.
  trace = rollout_warp(models["box"], trajectory)
  output["reference_box_metrics"] = validate(models["box"], trajectory, trace, config)
  samples = {phase: round(t / models["box"].opt.timestep) for phase, t in (("grip", 8.5), ("carry", 12.5))}
  snapshot_data = {}
  for phase, step in samples.items():
    snapshot_data[f"{phase}_qpos"] = trace["qpos"][step]
    snapshot_data[f"{phase}_qvel"] = trace["qvel"][step]
  np.savez_compressed(args.output.with_suffix(".snapshots.npz"), **snapshot_data)
  output["snapshot_times"] = {phase: (step + 1) * models["box"].opt.timestep for phase, step in samples.items()}
  sdf_trace = rollout_warp(models["sdf"], trajectory)
  output["diagnostic_metrics"] = {
    "box": output["reference_box_metrics"],
    "sdf": validate(models["sdf"], trajectory, sdf_trace, config),
  }
  with wp.ScopedDevice("cuda:0"):
    warp_models = {mode: mjw.put_model(model) for mode, model in models.items()}
    for nworld in args.worlds:
      for repeat in range(args.repeats):
        for mode in ("box", "sdf") if repeat % 2 == 0 else ("sdf", "box"):
          print(f"Measuring {mode}, {nworld} worlds, repeat {repeat + 1}/{args.repeats}", flush=True)
          result = full_task_measure(models[mode], warp_models[mode], trajectory, nworld, config["nconmax"], config["njmax"])
          # Cached allocations make process memory unsuitable for a per-mode comparison.
          result.pop("gpu_memory_mib")
          output["runs"].append({"finger_collision": mode, "repeat": repeat + 1, **result})
          write_report(args.output, output)
      for phase in samples:
        for mode in ("box", "sdf"):
          result = collision_measure(
            models[mode],
            warp_models[mode],
            snapshot_data[f"{phase}_qpos"],
            snapshot_data[f"{phase}_qvel"],
            nworld,
            config,
            args.repeats,
          )
          output["collision_runs"].append({"finger_collision": mode, "phase": phase, "nworld": nworld, **result})
          write_report(args.output, output)
  print(f"Saved {args.output} and {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
  main()

"""Interleave box, octree SDF and dense SDF runs under identical timing conditions."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import finger_cost_profile as profiler
import mujoco
import numpy as np
import warp as wp
from collision_cost_profile import STAGES
from collision_cost_profile import average
from dense_sdf import attach_dense_sdf
from finger_collision_benchmark import collision_measure
from grasp import build_model
from grasp import make_trajectory
from grasp import validate
from grasp_config import validate_config
from grasp_warp import rollout_warp
from parallel_efficiency_benchmark import full_task_measure

import mujoco_warp as mjw
from mujoco_warp._src.collision_sdf import DenseSDF
from mujoco_warp._src.types import BroadphaseType

MODES = ("box_octree", "sdf_octree", "sdf_dense")


def summarize(output):
  rows = []
  for nworld in output["worlds"]:
    for mode in MODES:
      runs = [r for r in output["full_task"] if r["mode"] == mode and r["nworld"] == nworld]
      if not runs:
        continue
      seconds = statistics.median(r["seconds"] for r in runs)
      rows.append(
        {
          "mode": mode,
          "nworld": nworld,
          "seconds": seconds,
          "ms_per_batch_step": seconds * 1000 / output["steps"],
          "world_steps_per_second": nworld * output["steps"] / seconds,
          "repeats_seconds": [r["seconds"] for r in runs],
          "placement_error_max_m": max(r["placement_error_max_m"] for r in runs),
        }
      )
  profiles = []
  for nworld in output["profile_worlds"]:
    for mode in MODES:
      runs = [r for r in output["profiles"] if r["mode"] == mode and r["nworld"] == nworld]
      split = sorted([r for r in runs if r["split"]], key=lambda r: r["average_ms"]["collision"])
      control = [r for r in runs if not r["split"]]
      if not split or not control:
        continue
      chosen = split[len(split) // 2]
      ms = dict(chosen["average_ms"])
      ms["other"] = ms["collision"] - ms["broadphase"] - ms["narrowphase"]
      control_ms = statistics.median(r["average_ms"]["collision"] for r in control)
      profiles.append(
        {
          "mode": mode,
          "nworld": nworld,
          "representative_repeat": chosen["repeat"],
          "ms": ms,
          "collision_percent": {k: ms[k] / ms["collision"] * 100 for k in ("broadphase", "narrowphase", "other")},
          "control_collision_ms": control_ms,
          "timing_difference_percent": (ms["collision"] / control_ms - 1) * 100,
        }
      )
  return {"full_task": rows, "profiles": profiles}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--worlds", type=int, nargs="+", default=[1, 512])
  parser.add_argument("--profile-worlds", type=int, nargs="*", default=[1])
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--smoke", action="store_true", help="One-step tool check; not a performance result")
  parser.add_argument("--output", type=Path, default=Path("contrib/piper_h/results/review_20260929/benchmark.json"))
  args = parser.parse_args()
  if args.repeats < 1 or any(n < 1 for n in args.worlds + args.profile_worlds):
    parser.error("worlds and repeats must be positive")
  if not wp.is_cuda_available():
    parser.error("CUDA required")
  wp.config.log_level = wp.LOG_WARNING
  args.output.parent.mkdir(parents=True, exist_ok=True)
  config = validate_config({})
  output = {
    "device": wp.get_device("cuda:0").name,
    "mujoco_version": mujoco.__version__,
    "warp_version": wp.__version__,
    "config": config,
    "worlds": args.worlds,
    "profile_worlds": args.profile_worlds,
    "repeats": args.repeats,
    "smoke": args.smoke,
    "full_task": [],
    "profiles": [],
    "frozen_collision": [],
    "validation": {},
    "started_unix": time.time(),
    "process_id": os.getpid(),
    "variants": {
      "box_octree": "box fingers + octree object",
      "sdf_octree": "octree fingers + octree object",
      "sdf_dense": "257^3 cell gradients: dense fingers + dense object",
    },
  }
  source_dir = Path(__file__).resolve().parent
  sources = [*source_dir.glob("*.py"), *source_dir.glob("*.xml"), source_dir / "meshes/cube.obj"]
  sources.append(Path(mjw.__file__).resolve().parent / "_src/collision_sdf.py")
  output["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)}

  def save():
    output["summary"] = summarize(output)
    args.output.write_text(json.dumps(output, indent=2) + "\n")

  with wp.ScopedDevice("cuda:0"):
    models = {"box": build_model({"finger_collision": "box"}), "sdf": build_model()}
    for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "qpos0", "actuator_gainprm"):
      np.testing.assert_array_equal(getattr(models["box"], name), getattr(models["sdf"], name))
    trajectory = make_trajectory(models["box"])
    np.testing.assert_allclose(trajectory["ctrl"], make_trajectory(models["sdf"])["ctrl"], atol=1e-12, rtol=0)
    if args.smoke:
      trajectory["ctrl"] = trajectory["ctrl"][:1]
    output["steps"] = len(trajectory["ctrl"])
    warp_models = {k: mjw.put_model(v) for k, v in models.items()}
    if any(m.opt.broadphase != BroadphaseType.NXN for m in warp_models.values()):
      raise RuntimeError("Profiler requires NXN broadphase")
    output["dense_cache"] = attach_dense_sdf(models["sdf"], warp_models["sdf"], 17 if args.smoke else 257)
    grids = warp_models["sdf"].dense_sdf

    def select(mode):
      key = "box" if mode == "box_octree" else "sdf"
      warp_models[key].dense_sdf = grids if mode == "sdf_dense" else DenseSDF()
      return models[key], warp_models[key]

    snapshots = {}
    if not args.smoke:
      for mode in MODES:
        model, warp_model = select(mode)
        trace = rollout_warp(model, trajectory, warp_model=warp_model)
        metrics = validate(model, trajectory, trace)
        output["validation"][mode] = metrics
        if mode == "box_octree":
          for phase, t in (("grip", 8.5), ("carry", 12.5)):
            index = round(t / model.opt.timestep)
            snapshots[phase] = (trace["qpos"][index].copy(), trace["qvel"][index].copy())
        save()
        if not metrics["passed"]:
          raise RuntimeError(f"{mode}: full grasp validation failed")
        print(f"Validation passed: {mode}", flush=True)
      np.savez_compressed(
        args.output.with_suffix(".snapshots.npz"),
        **{f"{phase}_{field}": values[i] for phase, values in snapshots.items() for i, field in enumerate(("qpos", "qvel"))},
      )

    # Interleave modes and rotate their order so each occupies each timing position.
    for nworld in args.worlds:
      for repeat in range(args.repeats):
        order = MODES[repeat % 3 :] + MODES[: repeat % 3]
        for mode in order:
          model, warp_model = select(mode)
          print(f"Full: {mode}, worlds={nworld}, repeat={repeat + 1}", flush=True)
          result = full_task_measure(model, warp_model, trajectory, nworld, config["nconmax"], config["njmax"])
          result.pop("gpu_memory_mib")  # Allocator/process memory is not per-variant storage.
          output["full_task"].append({"mode": mode, "repeat": repeat + 1, **result})
          save()
          if result["overflow"] or not result["finite_qpos"] or (not args.smoke and result["placement_error_max_m"] > 0.005):
            raise RuntimeError(f"{mode}: invalid timed rollout")

    previous_stages = profiler.STAGES
    try:
      for nworld in args.profile_worlds:
        for repeat in range(args.repeats):
          for mode in MODES[repeat % 3 :] + MODES[: repeat % 3]:
            model, warp_model = select(mode)
            # A separate total-only control measures event instrumentation effects.
            for split in (False, True) if repeat == 0 else (True,):
              profiler.STAGES = STAGES if split else STAGES[:1]
              print(f"Profile: {mode}, worlds={nworld}, repeat={repeat + 1}, split={split}", flush=True)
              result = profiler.measure(model, warp_model, trajectory, True, 20, config, nworld=nworld)
              output["profiles"].append(
                {"mode": mode, "repeat": repeat + 1, "split": split, "average_ms": average(result["samples"]), **result}
              )
              save()
              if (
                result["final_overflow"]
                or not result["finite_final_qpos"]
                or (not args.smoke and result["placement_error_max_m"] > 0.005)
              ):
                raise RuntimeError(f"{mode}: invalid profiled rollout")
    finally:
      profiler.STAGES = previous_stages

    for phase, (qpos, qvel) in snapshots.items():
      for nworld in args.worlds:
        for repeat in range(args.repeats):
          for mode in MODES[repeat % 3 :] + MODES[: repeat % 3]:
            model, warp_model = select(mode)
            result = collision_measure(model, warp_model, qpos, qvel, nworld, config, 1)
            output["frozen_collision"].append({"mode": mode, "nworld": nworld, "phase": phase, "repeat": repeat + 1, **result})
            save()
            if result["overflow"]:
              raise RuntimeError(f"{mode}: frozen collision capacity exceeded")
    output["finished_unix"] = time.time()
    save()
  print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
  main()

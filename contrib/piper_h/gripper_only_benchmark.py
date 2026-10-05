"""Paired full-arm / prescribed-gripper benchmarks, including trajectory preparation."""

import argparse
import gc
import json
import resource
import statistics
import time
from pathlib import Path
from unittest import mock

import finger_cost_profile as profiler
import grasp
import gripper_only as go
import mujoco
import numpy as np
import warp as wp
from dense_sdf import attach_dense_sdf
from grasp_config import validate_config
from grasp_dashboard import validate_run
from grasp_warp import rollout_warp
from parallel_efficiency_benchmark import gpu_memory_mib

import mujoco_warp as mjw


def measure(model, wm, trajectory, worlds, config, detailed=None):
  """Time the same one-step submission loop for both models, with no replay writes."""
  data = mujoco.MjData(model)
  data.qpos[:] = trajectory["qpos"][0]
  data.qvel[:] = trajectory["qvel"][0]
  mujoco.mj_forward(model, data)
  d = mjw.put_data(model, data, nworld=worlds, nconmax=config["nconmax"], njmax=config["njmax"])
  targets = wp.array(trajectory["ctrl"], dtype=float)
  index = wp.zeros(1, dtype=int)
  adapter = None
  original_step = mjw.step
  if "root_motion" in trajectory:
    adapter = go.PrescribedGripper(model, trajectory["root_motion"])
    adapter.state = go.motion_state(wm, d)

  def advance(m, state):
    if adapter is None:
      original_step(m, state)
    else:
      adapter.step(m, state, index)

  with mock.patch.object(mjw, "step", advance):
    graph, _ = profiler.capture_graph(wm, d, targets, index)
    sampled, events = (
      (None, {}) if detailed is None else profiler.capture_graph(wm, d, targets, index, detailed=detailed, timed=True)
    )
  for _ in range(10):
    wp.capture_launch(graph)
  wp.synchronize()
  wp.copy(d.qpos, wp.array(np.tile(data.qpos.astype(np.float32), (worlds, 1))))
  wp.copy(d.qvel, wp.array(np.tile(data.qvel.astype(np.float32), (worlds, 1))))
  d.qacc.zero_()
  d.qacc_warmstart.zero_()
  d.time.zero_()
  d.overflow.zero_()
  index.zero_()
  wp.synchronize()
  samples = []
  start = time.perf_counter()
  count = len(trajectory["ctrl"])
  for i in range(count):
    if sampled is not None and i % 100 == 0:
      wp.capture_launch(sampled)
      wp.synchronize_event(events["step"][1])
      samples.append(
        {
          "step": i,
          "weight": min(100, count - i),
          "phase": profiler.phase(i * model.opt.timestep * 24 / config["duration"]),
          "milliseconds": {key: wp.get_event_elapsed_time(*pair, synchronize=False) for key, pair in events.items()},
        }
      )
    else:
      wp.capture_launch(graph)
  wp.synchronize()
  elapsed = time.perf_counter() - start
  q = d.qpos.numpy()
  free = int(model.joint("cat_free").qposadr[0])
  errors = np.linalg.norm(q[:, free : free + 2] - grasp.PLACE, axis=1)
  result = {
    "seconds": elapsed,
    "world_steps_per_second": worlds * count / elapsed,
    "finite": bool(np.isfinite(q).all() and np.isfinite(d.qvel.numpy()).all()),
    "final_overflow": int(np.bitwise_or.reduce(d.overflow.numpy())),
    "placement_mean_m": float(errors.mean()),
    "placement_max_m": float(errors.max()),
    "placement_success_fraction": float(np.mean(errors <= config["placement_tolerance"])),
    "process_gpu_memory_mib": gpu_memory_mib(),
    "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
  }
  if detailed is not None:
    result.update(detailed=detailed, samples=samples, summary=profiler.average_samples(samples, detailed))
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--worlds", nargs="+", type=int, default=[1, 128, 512, 1024])
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/gripper_only/benchmark.json")
  parser.add_argument("--profile", action="store_true", help="Add calibrated 512-world CUDA stage samples")
  args = parser.parse_args()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  result = {"versions": {"warp": wp.__version__, "mujoco": mujoco.__version__}, "groups": []}

  def save():
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))

  for worlds in args.worlds:
    config = validate_config({"nworld": worlds})
    group = {"nworld": worlds, "config": config, "runs": [], "validation": {}, "preparation": {}}
    result["groups"].append(group)
    start = time.perf_counter()
    full = grasp.build_model(config)
    reduced = go.detach(config)
    trajectory = grasp.make_trajectory(full, config)
    group["preparation"]["models_and_targets_seconds"] = time.perf_counter() - start
    with wp.ScopedDevice("cuda:0"):
      # These complete recording runs supply reference trajectories and validate every step.
      start = time.perf_counter()
      trace = rollout_warp(full, trajectory, nworld=worlds, record_contact_forces=True, sdf_mode="dense")
      group["preparation"]["reference_rollout_seconds"] = time.perf_counter() - start
      group["validation"]["full_arm"] = validate_run(full, trajectory, trace, config)
      start = time.perf_counter()
      motion = go.export_motion(full, trace["qpos"], trace["qvel"])
      group["preparation"]["export_seconds"] = time.perf_counter() - start
      del trace
      gc.collect()
      reduced_trajectory = go.reduced_trajectory(full, reduced, trajectory, motion)
      trace = rollout_warp(reduced, reduced_trajectory, nworld=worlds, record_contact_forces=True, sdf_mode="dense")
      group["validation"]["gripper_only"] = validate_run(reduced, reduced_trajectory, trace, config)
      group["validation"]["gripper_only"]["physics_seconds"] = trace["physics_seconds"]
      del trace
      gc.collect()
      save()
      print("Validation", worlds, {key: value["passed_count"] for key, value in group["validation"].items()}, flush=True)
      if not all(value["passed"] for value in group["validation"].values()):
        raise RuntimeError("Paired correctness validation failed; performance measurement skipped")
      full_warp, reduced_warp = mjw.put_model(full), mjw.put_model(reduced)
      attach_dense_sdf(full, full_warp, 257)
      attach_dense_sdf(reduced, reduced_warp, 257)
      settings = {"full_arm": (full, full_warp, trajectory), "gripper_only": (reduced, reduced_warp, reduced_trajectory)}
      for repeat in range(args.repeats):
        for mode in ("full_arm", "gripper_only") if repeat % 2 == 0 else ("gripper_only", "full_arm"):
          row = measure(*settings[mode], worlds, config)
          row.update(mode=mode, repeat=repeat)
          group["runs"].append(row)
          print("Timing", worlds, mode, row["seconds"], flush=True)
          save()
          if not row["finite"] or row["final_overflow"] or row["placement_success_fraction"] < 1:
            raise RuntimeError("Timed run failed validation; do not interpret it as a speedup")
      group["median_seconds"] = {
        mode: statistics.median(r["seconds"] for r in group["runs"] if r["mode"] == mode) for mode in settings
      }
      if worlds == 512 and args.profile:
        group["profiles"] = []
        for repeat in range(3):
          for mode in settings:
            for detailed in (False, True):
              row = measure(*settings[mode], worlds, config, detailed)
              row.update(mode=mode, repeat=repeat)
              group["profiles"].append(row)
              print("Profile", mode, repeat, detailed, flush=True)
              save()
      save()
      del settings, full_warp, reduced_warp, reduced_trajectory, motion
      gc.collect()


if __name__ == "__main__":
  main()

"""Measure batched Warp throughput for the PiPER H full task or saved snapshots."""

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp
from grasp import PLACE
from grasp import build_model
from grasp import make_trajectory

import mujoco_warp as mjw

PHASES = {"settle": "0.5", "grip": "7.5", "carry": "12.5", "release": "18.5"}


@wp.kernel
def _set_control(
  # In:
  targets: wp.array2d[float],
  index: wp.array[int],
  # Data out:
  ctrl_out: wp.array2d[float],
):
  world, actuator = wp.tid()
  ctrl_out[world, actuator] = targets[index[0], actuator]


@wp.kernel
def _advance(
  # Out:
  index_out: wp.array[int],
):
  index_out[0] += 1


def gpu_memory_mib():
  result = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
    capture_output=True,
    text=True,
    check=False,
  )
  if result.returncode:
    return None
  for line in result.stdout.splitlines():
    pid, memory = (item.strip() for item in line.split(",", 1))
    if int(pid) == os.getpid():
      return int(memory)
  return None


def load_snapshots(path):
  archive = np.load(path)
  return {
    phase: {field: archive[f"{time_key}_{field}"] for field in ("qpos", "qvel", "ctrl", "time", "ncon")}
    for phase, time_key in PHASES.items()
  }


def make_data(model, sample):
  data = mujoco.MjData(model)
  data.qpos[:] = sample["qpos"]
  data.qvel[:] = sample["qvel"]
  data.ctrl[:] = sample["ctrl"]
  data.time = float(sample["time"])
  mujoco.mj_forward(model, data)
  return data


def cpu_measure(model, sample, steps, repeats):
  elapsed = []
  for _ in range(repeats):
    data = make_data(model, sample)
    start = time.perf_counter()
    for _ in range(steps):
      mujoco.mj_step(model, data)
    elapsed.append(time.perf_counter() - start)
  return {"median_seconds": statistics.median(elapsed), "repeats_seconds": elapsed}


def warp_measure(model, warp_model, samples, nworld, steps, repeats, nconmax, njmax):
  data = make_data(model, samples["grip"])
  warp_data = mjw.put_data(model, data, nworld=nworld, nconmax=nconmax, njmax=njmax)
  zero_warmstart = wp.zeros((nworld, model.nv), dtype=float)
  with wp.ScopedCapture() as capture:
    mjw.step(warp_model, warp_data)
  graph = capture.graph
  memory = gpu_memory_mib()
  results = {}
  for phase, sample in samples.items():
    qpos = wp.array(np.tile(sample["qpos"].astype(np.float32), (nworld, 1)))
    qvel = wp.array(np.tile(sample["qvel"].astype(np.float32), (nworld, 1)))
    ctrl = wp.array(np.tile(sample["ctrl"].astype(np.float32), (nworld, 1)))
    initial_time = wp.array(np.full(nworld, float(sample["time"]), dtype=np.float32))

    def reset():
      wp.copy(warp_data.qpos, qpos)
      wp.copy(warp_data.qvel, qvel)
      wp.copy(warp_data.ctrl, ctrl)
      wp.copy(warp_data.time, initial_time)
      wp.copy(warp_data.qacc_warmstart, zero_warmstart)
      wp.synchronize()

    reset()
    for _ in range(10):
      wp.capture_launch(graph)
    wp.synchronize()

    elapsed = []
    overflow = 0
    final_contacts = []
    for _ in range(repeats):
      reset()
      start = time.perf_counter()
      for _ in range(steps):
        wp.capture_launch(graph)
      wp.synchronize()
      elapsed.append(time.perf_counter() - start)
      overflow |= int(np.max(warp_data.overflow.numpy()))
      final_contacts.append(float(np.sum(warp_data.nacon.numpy()) / nworld))
    results[phase] = {
      "median_seconds": statistics.median(elapsed),
      "repeats_seconds": elapsed,
      "overflow": overflow,
      "final_contacts_per_world": statistics.mean(final_contacts),
      "initial_contacts": int(sample["ncon"]),
    }
    print(
      f"{nworld:4d} worlds {phase:7s}: {results[phase]['median_seconds']:.4f}s / {steps} steps "
      f"(overflow={overflow}, final contacts/world={results[phase]['final_contacts_per_world']:.1f})",
      flush=True,
    )
  return results, memory


def full_task_measure(model, warp_model, trajectory, nworld, nconmax, njmax):
  data = mujoco.MjData(model)
  data.qpos[:] = trajectory["qpos"][0]
  data.qvel[:] = trajectory["qvel"][0]
  mujoco.mj_forward(model, data)
  warp_data = mjw.put_data(model, data, nworld=nworld, nconmax=nconmax, njmax=njmax)
  targets = wp.array(trajectory["ctrl"], dtype=float)
  index = wp.zeros(1, dtype=int)
  with wp.ScopedCapture() as capture:
    wp.launch(_set_control, dim=(nworld, model.nu), inputs=[targets, index, warp_data.ctrl])
    mjw.step(warp_model, warp_data)
    wp.launch(_advance, dim=1, inputs=[index])
  graph = capture.graph
  memory = gpu_memory_mib()

  initial_qpos = wp.array(np.tile(trajectory["qpos"][0].astype(np.float32), (nworld, 1)))
  initial_qvel = wp.array(np.tile(trajectory["qvel"][0].astype(np.float32), (nworld, 1)))
  initial_time = wp.zeros(nworld, dtype=float)
  zero_warmstart = wp.zeros((nworld, model.nv), dtype=float)
  zero_index = wp.zeros(1, dtype=int)
  for _ in range(10):
    wp.capture_launch(graph)
  wp.synchronize()
  wp.copy(warp_data.qpos, initial_qpos)
  wp.copy(warp_data.qvel, initial_qvel)
  wp.copy(warp_data.time, initial_time)
  wp.copy(warp_data.qacc_warmstart, zero_warmstart)
  wp.copy(index, zero_index)
  wp.synchronize()

  start = time.perf_counter()
  for _ in range(len(trajectory["ctrl"])):
    wp.capture_launch(graph)
  wp.synchronize()
  elapsed = time.perf_counter() - start
  qpos = warp_data.qpos.numpy()
  free = model.joint("cat_free").qposadr[0]
  placement_errors = np.linalg.norm(qpos[:, free : free + 2] - PLACE, axis=1)
  result = {
    "nworld": nworld,
    "seconds": elapsed,
    "steps_per_world": len(trajectory["ctrl"]),
    "world_steps_per_second": nworld * len(trajectory["ctrl"]) / elapsed,
    "gpu_memory_mib": memory,
    "overflow": int(np.max(warp_data.overflow.numpy())),
    "finite_qpos": bool(np.isfinite(qpos).all()),
    "placement_error_mean_m": float(np.mean(placement_errors)),
    "placement_error_max_m": float(np.max(placement_errors)),
    "placement_error_p95_m": float(np.percentile(placement_errors, 95)),
    "placement_success_fraction": float(np.mean(placement_errors <= 0.05)),
  }
  print(
    f"{nworld:4d} worlds full trajectory: {elapsed:.2f}s, "
    f"{result['world_steps_per_second']:,.0f} world-steps/s, "
    f"placement pass={result['placement_success_fraction']:.1%}, "
    f"error mean/max={result['placement_error_mean_m']:.3f}/{result['placement_error_max_m']:.3f}m, "
    f"overflow={result['overflow']}",
    flush=True,
  )
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--snapshots", type=Path, help="Snapshot archive for --mode snapshots")
  parser.add_argument("--mode", choices=("snapshots", "full"), default="full")
  parser.add_argument("--worlds", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 1])
  parser.add_argument("--steps", type=int, default=200)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--nconmax", type=int, default=256)
  parser.add_argument("--njmax", type=int, default=1024)
  parser.add_argument("--output", type=Path, default=Path("/tmp/piper_h_parallel_results.json"))
  args = parser.parse_args()

  if args.mode == "snapshots" and args.snapshots is None:
    parser.error("--snapshots is required for --mode snapshots")

  if not wp.is_cuda_available():
    raise RuntimeError("CUDA device required")
  wp.config.log_level = wp.LOG_WARNING
  model = build_model()
  if args.mode == "full":
    trajectory = make_trajectory(model)
    output = {
      "mode": "full",
      "device": wp.get_device("cuda:0").name,
      "model_timestep": model.opt.timestep,
      "control_steps": len(trajectory["ctrl"]),
      "nconmax": args.nconmax,
      "njmax": args.njmax,
      "warp": [],
    }
    with wp.ScopedDevice("cuda:0"):
      warp_model = mjw.put_model(model)
      for nworld in args.worlds:
        output["warp"].append(full_task_measure(model, warp_model, trajectory, nworld, args.nconmax, args.njmax))
        args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)
    return

  samples = load_snapshots(args.snapshots)
  output = {
    "device": wp.get_device("cuda:0").name,
    "model_timestep": model.opt.timestep,
    "steps_per_run": args.steps,
    "repeats": args.repeats,
    "nconmax": args.nconmax,
    "njmax": args.njmax,
    "snapshots": {phase: float(sample["time"]) for phase, sample in samples.items()},
    "cpu": {phase: cpu_measure(model, sample, args.steps, args.repeats) for phase, sample in samples.items()},
    "warp": [],
  }
  print("CPU medians:", {phase: round(value["median_seconds"], 4) for phase, value in output["cpu"].items()}, flush=True)
  with wp.ScopedDevice("cuda:0"):
    warp_model = mjw.put_model(model)
    for nworld in args.worlds:
      phases, memory = warp_measure(model, warp_model, samples, nworld, args.steps, args.repeats, args.nconmax, args.njmax)
      output["warp"].append({"nworld": nworld, "gpu_memory_mib": memory, "phases": phases})
      args.output.write_text(json.dumps(output, indent=2) + "\n")
  print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
  main()

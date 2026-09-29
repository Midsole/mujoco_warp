"""Compare complete grasp workloads using a common runtime and submission cadence."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from threading import Thread

# Physim's environment supplies Newton/Warp; append this project's site-packages
# only for dependencies absent there (MuJoCo, absl, etils). Existing packages win.
REPO = Path(__file__).resolve().parents[2]
for site in sorted((REPO / ".venv/lib").glob("python*/site-packages")):
  sys.path.append(str(site))
sys.path.insert(0, str(REPO))

import mujoco
import numpy as np
import warp as wp
from dense_sdf import attach_dense_sdf
from grasp import PLACE
from grasp import build_model
from grasp import make_trajectory
from grasp import validate
from grasp_config import timing_settings
from grasp_config import validate_config
from grasp_warp import _advance
from grasp_warp import _control
from grasp_warp import rollout_warp

import mujoco_warp as mjw
from mujoco_warp._src.collision_sdf import DenseSDF


@contextmanager
def gpu_monitor(output):
  """Sample process utilization without requiring exclusive GPU ownership."""
  stop = Event()

  def sample():
    with (output / "gpu_processes.log").open("w") as stream:
      while not stop.is_set():
        try:
          result = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "um"], capture_output=True, text=True, timeout=5)
          stream.write(f"unix={time.time():.3f}\n{result.stdout}{result.stderr}\n")
          stream.flush()
        except (OSError, subprocess.TimeoutExpired) as error:
          stream.write(str(error) + "\n")
        stop.wait(3)

  thread = Thread(target=sample, daemon=True)
  thread.start()
  try:
    yield
  finally:
    stop.set()
    thread.join(timeout=6)


def task_trajectory(model, duration, control_hz):
  """Resample the entire native 24 s task and hold targets between control ticks."""
  native = make_trajectory(model)
  settings = timing_settings({"timestep": model.opt.timestep, "duration": duration, "control_hz": control_hz})
  steps = settings["physics_steps"]
  times = np.arange(steps) * model.opt.timestep
  stride = settings["control_steps"]
  control_times = (np.arange(steps) // stride) * stride * model.opt.timestep
  phase_times = control_times * 24 / settings["duration"]
  controls = np.column_stack([np.interp(phase_times, native["times"], c) for c in native["ctrl"].T])
  return {**native, **settings, "ctrl": controls, "times": times, "phase_times": times * 24 / settings["duration"]}


def measure_mujoco(model, warp_model, trajectory, nworld, block_steps):
  data = mujoco.MjData(model)
  data.qpos[:] = trajectory["qpos"][0]
  data.qvel[:] = trajectory["qvel"][0]
  mujoco.mj_forward(model, data)
  d = mjw.put_data(model, data, nworld=nworld, nconmax=256, njmax=1024)
  targets = wp.array(trajectory["ctrl"], dtype=float)
  index = wp.zeros(1, dtype=int)
  steps = len(trajectory["ctrl"])
  if steps % block_steps:
    raise ValueError("Task length must be divisible by graph length")
  # Warm up an ordinary step, then restore with a fresh data upload. CUDA capture
  # does not execute the captured device work. No timed state/force recording.
  wp.launch(_control, dim=(nworld, model.nu), inputs=[targets, index, d.ctrl])
  mjw.step(warp_model, d)
  wp.synchronize()
  d = mjw.put_data(model, data, nworld=nworld, nconmax=256, njmax=1024)
  with wp.ScopedCapture() as capture:
    for _ in range(block_steps):
      wp.launch(_control, dim=(nworld, model.nu), inputs=[targets, index, d.ctrl])
      mjw.step(warp_model, d)
      wp.launch(_advance, dim=1, inputs=[index])
  wp.synchronize()
  start = time.perf_counter()
  for _ in range(steps // block_steps):
    wp.capture_launch(capture.graph)
  wp.synchronize()
  seconds = time.perf_counter() - start
  qpos, qvel = d.qpos.numpy(), d.qvel.numpy()
  free = model.joint("cat_free").qposadr[0]
  error = np.linalg.norm(qpos[:, free : free + 2] - PLACE, axis=1)
  return {
    "seconds": seconds,
    "steps_per_world": steps,
    "finite": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all()),
    "overflow": int(np.bitwise_or.reduce(d.overflow.numpy())),
    "completed_worlds": int(np.sum(np.isfinite(qpos).all(axis=1))),
    "placement_error_max_mm": float(error.max() * 1000),
    "placement_pass_20mm_worlds": int(np.sum(error < 0.02)),
    "final_time_range_s": [float(d.time.numpy().min()), float(d.time.numpy().max())],
  }


# The final generated Physim adapter is saved alongside results. This replaces
# ONLY its outer recording/report loop: the captured step, diagnostics, contact
# candidate count, solver budgets, integration and termination guards are intact.
PHYSIM_TIMING_TAIL = """
    steps=int(round(duration/h));assert steps%nblock==0
    wp.synchronize();start=time.perf_counter()
    for block in range(steps//nblock):
        wp.capture_launch(capture.graph)
    wp.synchronize();seconds=time.perf_counter()-start
    final=states[0].body_q.numpy();vel=states[0].body_qd.numpy()
    met=metrics.numpy();ends=hard_end.numpy();pad=u.numpy()
    error=np.linalg.norm(final[:,:2]-np.array([0.,.075]),axis=1)
    complete=ends>=duration
    return dict(seconds=seconds,steps_per_world=steps,
        finite=bool(np.isfinite(final).all() and np.isfinite(vel).all()
                    and np.isfinite(met).all() and np.isfinite(pad).all()),
        overflow=None,completed_worlds=int(complete.sum()),
        terminated_times_s=ends.tolist(),placement_error_max_mm=float(error.max()*1000),
        placement_pass_20mm_worlds=int((complete & (error<.02)).sum()),
        lifted_worlds=int((met[:,1]>=0).sum()),dropped_worlds=int((met[:,2]>=0).sum()),
        max_height_range_mm=[float(met[:,0].min()*1000),float(met[:,0].max()*1000)],
        peak_total_soft_normal_load_n=float(met[:,6].max()),
        peak_compression_mm=float((pad[:,:,0]*np.tile([-1,1],B)[None,:]).max()*1000),
        joint_contact_audit=joint_stats.numpy().tolist(),final_q=final.tolist())
"""


def prepare_physim(source_root, output, block_steps):
  runtime = output / "physim_runtime"
  runtime.mkdir(parents=True, exist_ok=True)
  sources = [*source_root.glob("*.py"), *source_root.glob("core/*.py")]
  for source in sources:
    target = runtime / source.relative_to(source_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
  for name in ("models", "examples"):
    shutil.copytree(source_root / name, runtime / name, dirs_exist_ok=True)
  os.environ["LOCALAPPDATA"] = str(output / "generated")
  os.environ["GRASP_DISCRETIZATION"] = json.dumps(
    {"material_node_count": 49, "contact_sample_count": 441, "pad_layout": "global"}
  )
  sys.path.insert(0, str(runtime))
  import engine
  import projection_backend
  import trimesh
  from assets import import_stl
  from sdf import load_field

  # Export the same closed mesh; do not use Physim's differently sized built-in.
  cube = REPO / "contrib/piper_h/meshes/cube.obj"
  mesh = trimesh.load(cube, force="mesh")
  cube_stl = output / "cube_60mm.stl"
  mesh.export(cube_stl)
  meta = import_stl(cube_stl, unit="m", resolution=129, name="shared 60 mm cube")
  field, props, samples, meta = load_field(meta["id"], mode="sparse")
  original_specialize = projection_backend.specialize_rollout

  def timed_specialize(source):
    source = original_specialize(source)
    marker = "    steps=int(round(duration/h));assert steps%nblock==0"
    if source.count(marker) != 1 or source.count("    nblock=200") != 1:
      raise RuntimeError("Physim source layout changed; inspect adapter before benchmarking")
    original = source.replace("    nblock=200", f"    nblock={block_steps}")
    (output / "physim_adapter_original.py").write_text(original)
    timed = original[: original.index(marker)] + PHYSIM_TIMING_TAIL
    (output / "physim_adapter_timed.py").write_text(timed)
    return timed

  # Save a recording-enabled reference callable to validate measurement parity.
  original_run = projection_backend.make_run(
    engine.source, engine.namespace, "projected", subcells=True, joint=True, joint_budget="fast"
  )
  projection_backend.specialize_rollout = timed_specialize
  try:
    timed_run = projection_backend.make_run(
      engine.source, engine.namespace, "projected", subcells=True, joint=True, joint_budget="fast"
    )
  finally:
    projection_backend.specialize_rollout = original_specialize

  row = {
    "id": 0,
    "yaw": 0.0,
    "roll": 0.0,
    "pitch": 0.0,
    "height_offset_m": -0.005,
    "y_offset_m": 0.0,
    "compression_m": 0.0,
    "max_force_n": 10.0,
    "mu": 0.8,
    "grasp_height_mode": "table_safe",
    "tip_clearance_m": 0.0002,
  }

  def cases(nworld):
    return [{**row, "id": i} for i in range(nworld)]

  return timed_run, original_run, (field, props, samples), cases, meta, sources


def summarize(runs, duration):
  rows = []
  for worlds in sorted({r["nworld"] for r in runs}):
    for mode in ("mujoco_octree", "mujoco_dense", "physim_region"):
      group = [r for r in runs if r["nworld"] == worlds and r["mode"] == mode]
      if not group:
        continue
      seconds = statistics.median(r["seconds"] for r in group)
      steps = group[0]["steps_per_world"]
      rows.append(
        {
          "mode": mode,
          "nworld": worlds,
          "seconds": seconds,
          "ms_per_batch_step": seconds * 1000 / steps,
          "world_steps_per_second": worlds * steps / seconds,
          "realtime_factor": duration / seconds,
          "repeats_seconds": [r["seconds"] for r in group],
          "valid": all(
            r["finite"] and not r["overflow"] and r["completed_worlds"] == worlds and r["placement_pass_20mm_worlds"] == worlds
            for r in group
          ),
          "placement_error_max_mm": max(r["placement_error_max_mm"] for r in group),
        }
      )
  return rows


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--physim-root", type=Path, default=REPO.parent / "physim/grasp-lab")
  parser.add_argument("--worlds", type=int, nargs="+", default=[1, 128, 512])
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--duration", type=float, default=12.0)
  parser.add_argument("--timestep", type=float, default=0.0005)
  parser.add_argument("--control-hz", type=float, default=2000.0)
  parser.add_argument("--block-steps", type=int, default=200)
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument("--output", type=Path, default=REPO / "contrib/piper_h/results/physim_comparison")
  args = parser.parse_args()
  if args.repeats < 1 or any(w < 1 for w in args.worlds) or args.block_steps < 1 or args.block_steps % 2:
    parser.error("worlds and repeats must be positive; block-steps must be positive and even")
  if 24000 % args.block_steps or (args.smoke and args.block_steps != 200):
    parser.error("block-steps must divide 24000; smoke requires 200")
  # Physim phase scheduling and feedback currently update at every physics step.
  if args.duration != 12 or args.timestep != 0.0005 or args.control_hz != 2000:
    parser.error("This audited Physim adapter requires 12 s, 0.5 ms and 2000 Hz")
  args.output = args.output.resolve()
  args.output.mkdir(parents=True, exist_ok=True)
  if not wp.is_cuda_available():
    parser.error("CUDA required")
  if importlib.metadata.version("warp-lang") != "1.17.0":
    parser.error("Use Physim's existing uv Python environment for the common Warp 1.17 runtime")
  wp.config.log_level = wp.LOG_WARNING
  out = {
    "started_unix": time.time(),
    "process_id": os.getpid(),
    "smoke": args.smoke,
    "config": vars(args) | {"output": str(args.output), "physim_root": str(args.physim_root)},
    "device": wp.get_device("cuda:0").name,
    "versions": {p: importlib.metadata.version(p) for p in ("mujoco", "warp-lang", "numpy", "newton", "trimesh")},
    "module_paths": {"warp": wp.__file__, "mujoco": mujoco.__file__, "mujoco_warp": mjw.__file__, "numpy": np.__file__},
    "mujoco_config": validate_config({"timestep": args.timestep}),
    "runs": [],
    "validation": {},
  }

  def save():
    out["summary"] = summarize(out["runs"], args.duration)
    (args.output / "benchmark.json").write_text(json.dumps(out, indent=2) + "\n")

  with gpu_monitor(args.output), wp.ScopedDevice("cuda:0"):
    model = build_model({"timestep": args.timestep})
    trajectory = task_trajectory(model, args.duration, args.control_hz)
    warp_model = mjw.put_model(model)
    out["dense_cache"] = attach_dense_sdf(model, warp_model, 17 if args.smoke else 257)
    dense = warp_model.dense_sdf
    timed_run, reference_run, fields, make_cases, meta, physim_sources = prepare_physim(
      args.physim_root, args.output, args.block_steps
    )
    out["physim_asset"] = meta
    out["mujoco_model"] = {name: int(getattr(model, name)) for name in ("nbody", "ngeom", "nv", "nu", "nq")}
    sources = [*Path(__file__).parent.glob("*.py"), *Path(__file__).parent.glob("*.xml"), *physim_sources]
    sources += [REPO / "mujoco_warp/_src/collision_sdf.py", Path(__file__).parent / "meshes/cube.obj"]
    sources += [args.physim_root / "models/piper-h" / name for name in ("collision-fields.npz", "inner-pad.json")]
    sources += [Path(__file__).parent / "meshes" / f"gripper_link{i}.stl" for i in (1, 2)]
    out["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    out["commits"] = {
      str(root): subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
      for root in (REPO, args.physim_root)
    }
    (args.output / "nvidia_smi_start.txt").write_text(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    modes = ("mujoco_octree", "mujoco_dense", "physim_region")
    if not args.smoke:
      for mode in modes[:2]:
        warp_model.dense_sdf = dense if mode == "mujoco_dense" else DenseSDF()
        trace = rollout_warp(model, trajectory, warp_model=warp_model)
        out["validation"][mode] = validate(model, trajectory, trace)
        np.savez_compressed(args.output / f"{mode}_validation.npz", **trace, phase_time=trajectory["phase_times"])
        save()
        print(f"Validation {mode}: {out['validation'][mode]}", flush=True)
      report, history, q0 = reference_run(*fields, make_cases(1), 1e-5, 0, mass=0.1)
      out["validation"]["physim_region"] = report
      np.savez_compressed(args.output / "physim_validation.npz", q0=q0, **history)
      out["validation"]["physim_region"]["finite_history"] = bool(all(np.isfinite(values).all() for values in history.values()))
      save()
    for worlds in args.worlds:
      for repeat in range(args.repeats):
        for mode in modes[repeat % 3 :] + modes[: repeat % 3]:
          print(f"Timing {mode}, worlds={worlds}, repeat={repeat + 1}", flush=True)
          setup_start = time.perf_counter()
          if mode == "physim_region":
            result = timed_run(*fields, make_cases(worlds), 1e-5, 0, mass=0.1, duration=0.1 if args.smoke else 12.0)
          else:
            warp_model.dense_sdf = dense if mode == "mujoco_dense" else DenseSDF()
            task = {**trajectory, "ctrl": trajectory["ctrl"][:200]} if args.smoke else trajectory
            result = measure_mujoco(model, warp_model, task, worlds, args.block_steps)
          out["runs"].append(
            {
              "mode": mode,
              "nworld": worlds,
              "repeat": repeat + 1,
              "setup_and_run_s": time.perf_counter() - setup_start,
              **result,
            }
          )
          if not args.smoke and worlds == 1 and mode == "physim_region":
            out["physim_recording_parity_max_difference"] = float(
              np.max(np.abs(np.asarray(result["final_q"]) - history["q"][-1]))
            )
            if out["physim_recording_parity_max_difference"] > 1e-6:
              save()
              raise RuntimeError("Removing replay recording changed Physim final state")
          save()
          print(
            json.dumps({k: result[k] for k in ("seconds", "finite", "completed_worlds", "placement_error_max_mm")}),
            flush=True,
          )
          if not result["finite"] or result["overflow"]:
            raise RuntimeError("Invalid timed simulation; saved raw result")
    out["finished_unix"] = time.time()
    out["source_hashes_unchanged"] = all(
      hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in out["source_sha256"].items()
    )
    save()
  print(f"Saved {args.output / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
  main()

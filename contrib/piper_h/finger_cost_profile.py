"""Sample CUDA graph stage timings for single-world box/SDF finger trajectories."""

import argparse
import contextlib
import functools
import json
import time
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np
import warp as wp
from grasp import PLACE
from grasp import build_model
from grasp import make_trajectory
from grasp_config import validate_config
from parallel_efficiency_benchmark import _advance
from parallel_efficiency_benchmark import _set_control

import mujoco_warp as mjw
from mujoco_warp._src import collision_driver
from mujoco_warp._src import constraint
from mujoco_warp._src import forward
from mujoco_warp._src import solver

STAGES = (
  (forward, "fwd_position", "position"),
  (collision_driver, "collision", "collision"),
  (collision_driver, "sdf_narrowphase", "sdf_contacts"),
  (constraint, "make_constraint", "constraints"),
  (forward, "fwd_velocity", "velocity"),
  (forward, "fwd_actuation", "actuation"),
  (forward, "fwd_acceleration", "acceleration"),
  (solver, "solve", "solver"),
  (forward, "implicit", "integration"),
)


def phase(t):
  for end, name in ((6, "approach"), (8, "grip"), (14, "carry"), (19.5, "lower_release")):
    if t < end:
      return name
  return "retreat_settle"


def wrap_timer(fn, pair):
  @functools.wraps(fn)
  def wrapped(*args, **kwargs):
    wp.record_event(pair[0], external=True)
    result = fn(*args, **kwargs)
    wp.record_event(pair[1], external=True)
    return result

  return wrapped


def capture_graph(model, data, controls, index, detailed=False, timed=False):
  """Keep the normal step intact; patch only call wrappers during graph capture."""
  events = {name: (wp.Event(enable_timing=True), wp.Event(enable_timing=True)) for _, _, name in STAGES} if detailed else {}
  if timed:
    events["step"] = (wp.Event(enable_timing=True), wp.Event(enable_timing=True))
  with contextlib.ExitStack() as stack:
    if detailed:
      for module, name, label in STAGES:
        stack.enter_context(mock.patch.object(module, name, wrap_timer(getattr(module, name), events[label])))
    with wp.ScopedCapture() as capture:
      wp.launch(_set_control, dim=(data.nworld, controls.shape[1]), inputs=[controls, index, data.ctrl])
      if timed:
        wp.record_event(events["step"][0], external=True)
      mjw.step(model, data)
      if timed:
        wp.record_event(events["step"][1], external=True)
      wp.launch(_advance, dim=1, inputs=[index])
  return capture.graph, events


def measure(model, warp_model, trajectory, detailed, stride, config, *, nworld=1):
  if not len(trajectory["ctrl"]) or stride < 1 or nworld < 1:
    raise ValueError("trajectory must be nonempty; stride and nworld must be positive")
  initial = mujoco.MjData(model)
  initial.qpos[:] = trajectory["qpos"][0]
  initial.qvel[:] = trajectory["qvel"][0]
  mujoco.mj_forward(model, initial)
  data = mjw.put_data(model, initial, nworld=nworld, nconmax=config["nconmax"], njmax=config["njmax"])
  controls = wp.array(trajectory["ctrl"], dtype=float)
  index = wp.zeros(1, dtype=int)
  plain, _ = capture_graph(warp_model, data, controls, index)
  sampled, events = capture_graph(warp_model, data, controls, index, detailed=detailed, timed=True)
  for step in range(min(10, len(trajectory["ctrl"]))):
    wp.capture_launch(plain if step % 2 == 0 else sampled)
  wp.synchronize()
  # Match the reset used by the throughput benchmark; the model has no activation states.
  wp.copy(data.qpos, wp.array(np.tile(trajectory["qpos"][0].astype(np.float32), (nworld, 1))))
  wp.copy(data.qvel, wp.array(np.tile(trajectory["qvel"][0].astype(np.float32), (nworld, 1))))
  data.time.zero_()
  data.qacc_warmstart.zero_()
  index.zero_()
  wp.synchronize()
  samples = []
  started = time.perf_counter()
  for step in range(len(trajectory["ctrl"])):
    if step % stride:
      wp.capture_launch(plain)
      continue
    wp.capture_launch(sampled)
    wp.synchronize_event(events["step"][1])
    durations = {key: wp.get_event_elapsed_time(*pair, synchronize=False) for key, pair in events.items()}
    if any(not np.isfinite(value) or value < 0 for value in durations.values()):
      raise RuntimeError(f"Invalid CUDA event timing: {durations}")
    samples.append(
      {
        "step": step,
        "time": (step + 1) * model.opt.timestep,
        "phase": phase(step * model.opt.timestep),
        "weight": min(stride, len(trajectory["ctrl"]) - step),
        "milliseconds": durations,
      }
    )
  wp.synchronize()
  wall_seconds = time.perf_counter() - started
  free = model.joint("cat_free").qposadr[0]
  all_qpos = data.qpos.numpy()
  final_qpos = all_qpos[0]
  return {
    "nworld": nworld,
    "detailed": detailed,
    "wall_seconds_including_sampling": wall_seconds,
    "finite_final_qpos": bool(np.all(np.isfinite(all_qpos))),
    "placement_error_max_m": float(np.linalg.norm(all_qpos[:, free : free + 2] - PLACE, axis=1).max()),
    "final_qpos": final_qpos.tolist(),
    "final_object_position": final_qpos[free : free + 3].tolist(),
    "final_overflow": int(np.bitwise_or.reduce(data.overflow.numpy())),
    "samples": samples,
  }


LABELS = {
  "collision": "碰撞检测（含八叉树 SDF 查询）",
  "constraints": "约束构建",
  "solver": "约束求解器",
  "dynamics": "运动学与其他动力学",
  "integration": "隐式积分",
  "other": "其他 / 阶段间操作",
}
PHASE_LABELS = {
  "approach": "落稳 / 接近（0–6 s）",
  "grip": "夹紧（6–8 s）",
  "carry": "抬升 / 横移（8–14 s）",
  "lower_release": "下降 / 释放（14–19.5 s）",
  "retreat_settle": "退回 / 静置（19.5–24 s）",
}


def exclusive_times(raw):
  """Partition inclusive spans into disjoint categories without double counting."""
  values = {key: raw[key] for key in ("collision", "constraints", "solver", "integration")}
  values["dynamics"] = (
    raw["position"] - raw["collision"] - raw["constraints"] + raw["velocity"] + raw["actuation"] + raw["acceleration"]
  )
  values["other"] = raw["step"] - sum(values.values())
  return values


def average_samples(samples, detailed):
  weights = np.array([sample["weight"] for sample in samples])
  raw = {
    key: float(np.average([s["milliseconds"][key] for s in samples], weights=weights)) for key in samples[0]["milliseconds"]
  }
  result = {"gpu_step_ms": raw["step"], "samples": len(samples), "represented_steps": int(weights.sum())}
  if detailed:
    result["stage_ms"] = exclusive_times(raw)
    result["stage_percent"] = {key: value / raw["step"] * 100 for key, value in result["stage_ms"].items()}
    result["sdf_narrowphase_ms"] = raw["sdf_contacts"]
  return result


def write_summary(path, output):
  """Save additive mean stage costs, phase breakdowns and event-overhead calibration."""
  summary = {"modes": {}, "nworld": output["nworld"], "steps": output["steps"], "stride": output["stride"]}
  for mode in ("box", "sdf"):
    runs = [run for run in output["runs"] if run["mode"] == mode]
    detailed_runs = sorted(
      [run for run in runs if run["detailed"]],
      key=lambda run: average_samples(run["samples"], True)["gpu_step_ms"],
    )
    representative = detailed_runs[len(detailed_runs) // 2]
    samples = representative["samples"]
    result = average_samples(samples, True)
    result["representative_repeat"] = representative["repeat"]
    result["pooled_all_repeats"] = average_samples([s for r in detailed_runs for s in r["samples"]], True)
    result["outer_only_gpu_step_ms"] = float(
      np.median([average_samples(run["samples"], False)["gpu_step_ms"] for run in runs if not run["detailed"]])
    )
    result["timing_overhead_percent"] = (result["gpu_step_ms"] / result["outer_only_gpu_step_ms"] - 1) * 100
    result["phases"] = {
      name: average_samples([s for s in samples if s["phase"] == name], True)
      for name in PHASE_LABELS
      if any(s["phase"] == name for s in samples)
    }
    result["per_repeat"] = [
      {"repeat": run["repeat"], "detailed": run["detailed"], **average_samples(run["samples"], run["detailed"])} for run in runs
    ]
    result["final_object_position_range_m"] = np.ptp([run["final_object_position"] for run in runs], axis=0).tolist()
    summary["modes"][mode] = result
  path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  box, sdf = (summary["modes"][mode] for mode in ("box", "sdf"))
  delta_step = sdf["gpu_step_ms"] - box["gpu_step_ms"]
  constraint_savings = sum(box["stage_ms"][key] - sdf["stage_ms"][key] for key in ("constraints", "solver"))
  lines = [
    "# PiPER 单环境 GPU 耗时占比：盒体手指与 SDF 手指",
    "",
    f"设备：{output['device']}；MuJoCo {output['mujoco_version']}，Warp {output['warp_version']}。",
    "",
    "## 方法与计时口径",
    "",
    "- 两组均为一个环境、同一 60 mm 方块八叉树 SDF、相同控制轨迹及物理参数；只切换手指几何。",
    f"- 每次 {output['steps']:,} 步，步长 1 ms；每 {output['stride']} 步采样一次，两组各重复 {output['repeats']} 次。",
    "- 在原 CUDA Graph 内用 CUDA events 标记阶段边界，不拆分或替换物理步。",
    "- 另跑只记录物理步起止时间的对照，估计分项标记的开销；两类计时交替运行。",
    "- 每轮按代表步数加权求平均 GPU 每步耗时；采用总耗时中位数对应的整轮作为代表，分项不跨轮拼接。",
    "- 占比以该代表轮的 GPU 物理步总时间为分母。保留全部重复及合并平均，便于检查波动。",
    "- 分项互不重叠：运动学/动力学已扣除其内部的碰撞与约束构建；SDF 窄相是碰撞子项，不能再次相加。",
    "- 不包含 CPU 图提交、控制目标写入、模型编译、上传、回放和采样读取时间；不能直接当作端到端耗时占比。",
    "- 两组运动会因真实几何差异略有不同，反映相同任务下的实际成本；固定姿态对照见此前性能报告。",
    "",
    "## 完整轨迹平均分项（总耗时中位数对应轮次）",
    "",
    "| 阶段 | 盒体 ms/步 | 盒体占比 | SDF ms/步 | SDF 占比 | SDF − 盒体 ms/步 |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
  ]
  for key, label in LABELS.items():
    b, s = box["stage_ms"][key], sdf["stage_ms"][key]
    lines.append(
      f"| {label} | {b:.4f} | {box['stage_percent'][key]:.1f}% | {s:.4f} | {sdf['stage_percent'][key]:.1f}% | {s - b:+.4f} |"
    )
  lines += [
    f"| GPU 物理步合计 | {box['gpu_step_ms']:.4f} | 100% | {sdf['gpu_step_ms']:.4f} | 100% | {delta_step:+.4f} |",
    "",
    "## 结果解读",
    "",
    f"GPU 每步总时间增加 {(sdf['gpu_step_ms'] / box['gpu_step_ms'] - 1) * 100:.1f}%；"
    f"碰撞耗时为原来的 {sdf['stage_ms']['collision'] / box['stage_ms']['collision']:.2f} 倍。",
    f"碰撞增加 {sdf['stage_ms']['collision'] - box['stage_ms']['collision']:.4f} ms/步；"
    f"约束构建与求解合计减少 {constraint_savings:.4f} ms/步，抵消了部分增长。",
    "主要增加项是 SDF 窄相接触计算。此前相同携带姿态测试的接触点为盒体 64、SDF 8；"
    "这与约束成本下降相符，但不代表本轨迹每一步都保持这些接触点数。",
    "",
    "## 各动作阶段",
    "",
    "| 阶段 | 手指 | GPU ms/步 | 碰撞占比 | 约束构建占比 | 求解器占比 |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
  ]
  for phase_name, label in PHASE_LABELS.items():
    for mode in ("box", "sdf"):
      if phase_name not in summary["modes"][mode]["phases"]:
        continue
      result = summary["modes"][mode]["phases"][phase_name]
      pct = result["stage_percent"]
      lines.append(
        f"| {label} | {mode} | {result['gpu_step_ms']:.4f} | {pct['collision']:.1f}% | "
        f"{pct['constraints']:.1f}% | {pct['solver']:.1f}% |"
      )
  lines += ["", "## 计时开销校准", "", "| 手指 | 仅起止标记 ms/步 | 分项标记 ms/步 | 差异 |", "| --- | ---: | ---: | ---: |"]
  for mode in ("box", "sdf"):
    result = summary["modes"][mode]
    lines.append(
      f"| {mode} | {result['outer_only_gpu_step_ms']:.4f} | {result['gpu_step_ms']:.4f} | "
      f"{result['timing_overhead_percent']:+.2f}% |"
    )
  lines += ["", "分项计时的各轮平均 GPU ms/步：", ""]
  for mode in ("box", "sdf"):
    result = summary["modes"][mode]
    values = ", ".join(f"{r['gpu_step_ms']:.4f}" for r in result["per_repeat"] if r["detailed"])
    lines.append(f"- {mode}: {values}；代表轮次 {result['representative_repeat']}。")
  lines += [
    "",
    "此差异同时包含运行间波动，不是每个阶段的精确误差界限；小占比阶段尤其不宜过度解读。",
    "",
    f"SDF 窄相子项：盒体手指 {box['sdf_narrowphase_ms']:.4f} ms/步，SDF 手指 {sdf['sdf_narrowphase_ms']:.4f} ms/步。",
    "两组物体都使用八叉树 SDF，因此盒体手指版本同样有 SDF 窄相计算；这里包含桌面与物体等全部 SDF 接触。",
    "",
    "## 复现",
    "",
    "```bash",
    "uv run python contrib/piper_h/finger_cost_profile.py --repeats 3 --stride 20",
    "```",
  ]
  path.with_suffix(".md").write_text("\n".join(lines) + "\n")
  return summary


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--stride", type=int, default=20)
  parser.add_argument("--steps", type=int, help="Limit trajectory length for a smoke test")
  parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/cube_cost_profile/profile.json")
  args = parser.parse_args()
  if args.repeats < 1 or args.stride < 1 or (args.steps is not None and args.steps < 1):
    parser.error("repeats, stride and steps must be positive")
  wp.config.log_level = wp.LOG_WARNING
  config = validate_config({"object_shape": "cube"})
  models = {mode: build_model({**config, "finger_collision": mode}) for mode in ("box", "sdf")}
  trajectory = make_trajectory(models["box"])
  if args.steps is not None:
    trajectory["ctrl"] = trajectory["ctrl"][: args.steps]
  args.output.parent.mkdir(parents=True, exist_ok=True)
  output = {
    "config": config,
    "device": wp.get_device("cuda:0").name,
    "mujoco_version": mujoco.__version__,
    "warp_version": wp.__version__,
    "nworld": 1,
    "steps": len(trajectory["ctrl"]),
    "stride": args.stride,
    "repeats": args.repeats,
    "method": "CUDA events inside the existing step graph; sparse uniform sampling; outer-only calibration in separate runs",
    "runs": [],
  }
  with wp.ScopedDevice("cuda:0"):
    warp_models = {mode: mjw.put_model(model) for mode, model in models.items()}
    for repeat in range(args.repeats):
      for mode in ("box", "sdf") if repeat % 2 == 0 else ("sdf", "box"):
        for detailed in (False, True) if repeat % 2 == 0 else (True, False):
          print(f"{mode}, repeat {repeat + 1}, detailed={detailed}", flush=True)
          result = measure(models[mode], warp_models[mode], trajectory, detailed, args.stride, config)
          output["runs"].append({"mode": mode, "repeat": repeat + 1, **result})
          args.output.write_text(json.dumps(output, indent=2) + "\n")
          mean = np.average(
            [s["milliseconds"]["step"] for s in result["samples"]], weights=[s["weight"] for s in result["samples"]]
          )
          print(
            f"  sampled step={mean:.4f} ms, wall including sampling={result['wall_seconds_including_sampling']:.2f}s",
            flush=True,
          )

  write_summary(args.output, output)


if __name__ == "__main__":
  main()

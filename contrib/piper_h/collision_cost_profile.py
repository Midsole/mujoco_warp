"""Measure broadphase and narrowphase shares inside PiPER collision detection."""

import argparse
import json
from pathlib import Path

import finger_cost_profile as profiler
import mujoco
import numpy as np
import warp as wp
from grasp import build_model
from grasp import make_trajectory
from grasp_config import validate_config

import mujoco_warp as mjw
from mujoco_warp._src import collision_driver
from mujoco_warp._src.types import BroadphaseType

STAGES = (
  (collision_driver, "collision", "collision"),
  (collision_driver, "nxn_broadphase", "broadphase"),
  (collision_driver, "_narrowphase", "narrowphase"),
)


def average(samples):
  weights = [sample["weight"] for sample in samples]
  return {
    name: float(np.average([sample["milliseconds"][name] for sample in samples], weights=weights))
    for name in samples[0]["milliseconds"]
  }


def summarize(path, output):
  summary = {}
  for mode in ("box", "sdf"):
    runs = [run for run in output["runs"] if run["mode"] == mode]
    split = sorted([run for run in runs if run["split"]], key=lambda run: average(run["samples"])["collision"])
    representative = split[len(split) // 2]
    raw = average(representative["samples"])
    total = raw["collision"]
    stage = {name: raw[name] for name in ("broadphase", "narrowphase")}
    stage["other"] = total - sum(stage.values())
    control = float(np.median([average(run["samples"])["collision"] for run in runs if not run["split"]]))
    summary[mode] = {
      "representative_repeat": representative["repeat"],
      "collision_ms": total,
      "stage_ms": stage,
      "stage_percent": {name: value / total * 100 for name, value in stage.items()},
      "collision_only_timing_ms": control,
      "timing_difference_percent": (total / control - 1) * 100,
      "per_repeat": [{"repeat": run["repeat"], "split": run["split"], **average(run["samples"])} for run in runs],
      "phases": {
        phase: average([sample for sample in representative["samples"] if sample["phase"] == phase])
        for phase in profiler.PHASE_LABELS
        if any(sample["phase"] == phase for sample in representative["samples"])
      },
      "final_object_position_range_m": np.ptp([run["final_object_position"] for run in runs], axis=0).tolist(),
    }
  path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
  lines = [
    "# PiPER 两种手指：碰撞检测内部耗时",
    "",
    f"设备：{output['device']}；MuJoCo {output['mujoco_version']}，Warp {output['warp_version']}。",
    "",
    f"环境记录：{output.get('environment_note') or '未单独记录后台负载。'}",
    "",
    "## 方法",
    "",
    "- 一个环境，相同 60 mm 方块八叉树 SDF、控制轨迹和物理参数，仅切换盒体/SDF 手指。",
    f"- 每轮 {output['steps']:,} 步 × 1 ms；每 {output['stride']} 步采样一次，每组重复 {output['repeats']} 次。",
    "- 在原 CUDA Graph 内使用 CUDA events；粗筛为 nxn_broadphase，精细碰撞为整个 _narrowphase。",
    "- 采用碰撞总时间中位数对应的完整一轮；该轮按采样代表步数加权求平均，分项互不重叠。",
    "- 占比分母为 collision() 总 GPU 耗时。其他项包含计数器清零及阶段间操作，不包含约束构建和求解。",
    "- 另跑只标记 collision 起止的校准组，交替测量；编译、上传、回放和 CPU 提交不计入这些 GPU 时间。",
    "",
    "## 平均 GPU 耗时（ms/物理步）",
    "",
    "| 阶段 | 盒体 ms/步 | 盒体占比 | SDF ms/步 | SDF 占比 |",
    "| --- | ---: | ---: | ---: | ---: |",
  ]
  box, sdf = summary["box"], summary["sdf"]
  for name, label in (("broadphase", "粗筛"), ("narrowphase", "精细碰撞"), ("other", "清零及其他")):
    lines.append(
      f"| {label} | {box['stage_ms'][name]:.6f} | {box['stage_percent'][name]:.2f}% | "
      f"{sdf['stage_ms'][name]:.6f} | {sdf['stage_percent'][name]:.2f}% |"
    )
  lines += [f"| 合计 | {box['collision_ms']:.6f} | 100% | {sdf['collision_ms']:.6f} | 100% |", "", "## 计时对照", ""]
  for mode, values in summary.items():
    repeats = ", ".join(f"{run['collision']:.6f}" for run in values["per_repeat"] if run["split"])
    lines.append(
      f"- {mode}：仅总计时 {values['collision_only_timing_ms']:.6f} ms，分项计时 "
      f"{values['collision_ms']:.6f} ms，差异 {values['timing_difference_percent']:+.2f}%；三轮：{repeats} ms。"
    )
  lines += [
    "",
    "对照差异包含运行间波动，不是每个子项的误差界限；微秒级粗筛和其他项需保守解读。",
    "",
    "## 动作阶段分项",
    "",
    "| 阶段 | 手指 | 粗筛 ms/步 | 精细碰撞 ms/步 | 碰撞总计 ms/步 |",
    "| --- | --- | ---: | ---: | ---: |",
  ]
  for phase, label in profiler.PHASE_LABELS.items():
    for mode in ("box", "sdf"):
      if phase not in summary[mode]["phases"]:
        continue
      values = summary[mode]["phases"][phase]
      lines.append(
        f"| {label} | {mode} | {values['broadphase']:.6f} | {values['narrowphase']:.6f} | {values['collision']:.6f} |"
      )
  lines += ["", "```bash", "uv run python contrib/piper_h/collision_cost_profile.py --repeats 3 --stride 20", "```"]
  path.with_suffix(".md").write_text("\n".join(lines) + "\n")
  return summary


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--stride", type=int, default=20)
  parser.add_argument("--steps", type=int, help="Limit steps for a smoke test")
  parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/cube_collision_profile/profile.json")
  parser.add_argument("--note", default="", help="Record background load and timing conditions")
  args = parser.parse_args()
  if args.repeats < 1 or args.stride < 1 or (args.steps is not None and args.steps < 10):
    parser.error("repeats and stride must be positive; steps must be at least 10")
  wp.config.log_level = wp.LOG_WARNING
  config = validate_config({"object_shape": "cube"})
  models = {mode: build_model({**config, "finger_collision": mode}) for mode in ("box", "sdf")}
  trajectory = make_trajectory(models["box"])
  if args.steps is not None:
    trajectory["ctrl"] = trajectory["ctrl"][: args.steps]
  args.output.parent.mkdir(parents=True, exist_ok=True)
  output = {
    "config": config,
    "environment_note": args.note,
    "device": wp.get_device("cuda:0").name,
    "mujoco_version": mujoco.__version__,
    "warp_version": wp.__version__,
    "nworld": 1,
    "steps": len(trajectory["ctrl"]),
    "stride": args.stride,
    "repeats": args.repeats,
    "runs": [],
  }
  with wp.ScopedDevice("cuda:0"):
    warp_models = {mode: mjw.put_model(model) for mode, model in models.items()}
    if any(model.opt.broadphase != BroadphaseType.NXN for model in warp_models.values()):
      raise RuntimeError("This profile currently requires NXN broadphase")
    for repeat in range(args.repeats):
      for mode in ("box", "sdf") if repeat % 2 == 0 else ("sdf", "box"):
        for split in (False, True) if repeat % 2 == 0 else (True, False):
          profiler.STAGES = STAGES if split else STAGES[:1]
          print(f"{mode}, repeat {repeat + 1}, split={split}", flush=True)
          result = profiler.measure(models[mode], warp_models[mode], trajectory, True, args.stride, config)
          output["runs"].append({"mode": mode, "repeat": repeat + 1, "split": split, **result})
          args.output.write_text(json.dumps(output, indent=2) + "\n")
          print(f"  collision={average(result['samples'])['collision']:.6f} ms/step", flush=True)
  summarize(args.output, output)


if __name__ == "__main__":
  main()

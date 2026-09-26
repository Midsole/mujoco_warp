"""Run the existing viewer with frame-paced stepping for the grasp demonstration."""

import time

import mujoco
import mujoco.viewer

from mujoco_warp import viewer as shared_viewer


def run_native(model, data, step_fn):
  state = {"running": True, "step_once": False}

  def key_callback(key):
    if key == 32:
      state["running"] = not state["running"]
      print("继续播放" if state["running"] else "已暂停：按空格继续", flush=True)
    elif key == 46:
      state["step_once"] = True

  steps_per_frame = max(1, round(1 / (60 * model.opt.timestep)))
  frame_duration = max(1 / 60, steps_per_frame * model.opt.timestep)
  mujoco.mj_forward(model, data)
  print("开始抓取演示：空格暂停/继续，句号单步。前 1 秒仿真时间等待物体落稳。", flush=True)
  next_report = 2.0
  with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as window:
    while window.is_running():
      start = time.monotonic()
      steps = steps_per_frame if state["running"] else int(state["step_once"])
      state["step_once"] = False
      for _ in range(steps):
        step_fn(model, data)
      # Avoid recomputing CPU SDF contacts merely to display the GPU state.
      window.sync(state_only=True)
      if data.time >= next_report and next_report <= 24:
        print(f"抓取演示进度：{min(data.time, 24):.1f} / 24 秒仿真时间", flush=True)
        next_report += 2
      remaining = frame_duration - (time.monotonic() - start)
      if remaining > 0:
        time.sleep(remaining)


def main():
  # Only this example's subprocess uses the paced native loop. Keep the shared
  # loader, replay, CPU/Warp steppers and web viewer unchanged.
  shared_viewer._run_passive_viewer = run_native
  shared_viewer.main()


if __name__ == "__main__":
  main()

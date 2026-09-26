"""Check native playback pacing and pause controls without a display."""

from types import SimpleNamespace

import grasp_viewer
import pytest


@pytest.mark.parametrize("timestep", [0.001, 0.002])
def test_native_batches_steps_and_supports_pause(monkeypatch, timestep):
  data = SimpleNamespace(time=0.0)
  model = SimpleNamespace(opt=SimpleNamespace(timestep=timestep))
  frames = []
  callback = None

  class Window:
    def __enter__(self):
      return self

    def __exit__(self, *args):
      pass

    def is_running(self):
      return len(frames) < 4

    def sync(self, *, state_only):
      assert state_only
      frames.append(data.time)
      if len(frames) in (1, 3):
        callback(32)
      elif len(frames) == 2:
        callback(46)

  def launch(model, data, *, key_callback):
    nonlocal callback
    callback = key_callback
    return Window()

  def step(model, data):
    data.time += model.opt.timestep

  monkeypatch.setattr(grasp_viewer.mujoco.viewer, "launch_passive", launch)
  monkeypatch.setattr(grasp_viewer.mujoco, "mj_forward", lambda model, data: None)
  monkeypatch.setattr(grasp_viewer.time, "sleep", lambda seconds: None)
  grasp_viewer.run_native(model, data, step)
  batch = round(1 / (60 * timestep)) * timestep
  assert frames == pytest.approx([batch, batch, batch + timestep, 2 * batch + timestep])

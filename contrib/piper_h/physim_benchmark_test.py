"""Check the benchmark's task resampling and early-termination reporting."""

from types import SimpleNamespace

import numpy as np
import physim_benchmark as benchmark
import pytest


def test_task_resampling_preserves_full_motion_and_control_hold(monkeypatch):
  native = {
    "times": np.array([0.0, 24.0]),
    "ctrl": np.array([[0.0, 2.0], [24.0, 4.0]]),
    "duration": 24.0,
    "control_steps": 1,
    "effective_control_hz": 2000.0,
  }
  monkeypatch.setattr(benchmark, "make_trajectory", lambda model: native)
  model = SimpleNamespace(opt=SimpleNamespace(timestep=0.0005))
  trajectory = benchmark.task_trajectory(model, 12.0, 10.0)
  assert len(trajectory["ctrl"]) == 24000
  np.testing.assert_array_equal(trajectory["ctrl"][:200], np.tile([0.0, 2.0], (200, 1)))
  np.testing.assert_allclose(trajectory["ctrl"][200], [0.2, 2 + 0.2 / 12])
  np.testing.assert_allclose(trajectory["ctrl"][-1], [23.8, 2 + 23.8 / 12])
  assert trajectory["duration"] == 12.0
  assert trajectory["control_steps"] == 200
  assert trajectory["effective_control_hz"] == 10.0
  # Validation windows use the native phase clock, while physics still runs 12 s.
  assert trajectory["times"][-1] == 11.9995
  assert trajectory["phase_times"][-1] == 23.999


def test_task_control_frequency_is_capped_at_physics_frequency(monkeypatch):
  native = {"times": np.array([0.0, 24.0]), "ctrl": np.array([[0.0], [24.0]])}
  monkeypatch.setattr(benchmark, "make_trajectory", lambda model: native)
  model = SimpleNamespace(opt=SimpleNamespace(timestep=0.0005))
  trajectory = benchmark.task_trajectory(model, 12.0, 4000.0)
  assert trajectory["control_steps"] == 1
  assert trajectory["effective_control_hz"] == 2000.0


@pytest.mark.parametrize("completed,placed,valid", [(128, 128, True), (127, 128, False), (128, 127, False)])
def test_summary_requires_completion_and_placement_in_every_repeat(completed, placed, valid):
  runs = [
    {
      "mode": "physim_region",
      "nworld": 128,
      "seconds": seconds,
      "steps_per_world": 24000,
      "finite": True,
      "overflow": None,
      "completed_worlds": complete,
      "placement_pass_20mm_worlds": placed if seconds == 1.0 else 128,
      "placement_error_max_mm": 1.0 if placed == 128 else 30.0,
    }
    for seconds, complete in [(2.0, 128), (1.0, completed), (3.0, 128)]
  ]
  result = benchmark.summarize(runs, 12)[0]
  assert result["seconds"] == 2.0
  assert result["ms_per_batch_step"] == 1 / 12
  assert result["world_steps_per_second"] == 1536000
  assert result["valid"] is valid

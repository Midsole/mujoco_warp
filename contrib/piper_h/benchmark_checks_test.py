"""Regression checks for performance measurement validity."""

from types import SimpleNamespace

import numpy as np
import pytest
from finger_collision_benchmark import collision_overflow
from finger_cost_profile import measure
from parallel_efficiency_benchmark import full_task_measure

from mujoco_warp._src.types import OverflowType


@pytest.mark.parametrize("contacts,pairs", [(4, 4), (5, 4), (4, 5), (5, 5)])
def test_collision_only_capacity_check(contacts, pairs):
  def array(values):
    return SimpleNamespace(numpy=lambda: np.array(values))

  data = SimpleNamespace(naconmax=4, nacon=array([contacts]), ncollision=array([pairs]), overflow=array([0, 0]))
  expected = 0
  if contacts > 4:
    expected |= int(OverflowType.NARROWPHASE)
  if pairs > 4:
    expected |= int(OverflowType.BROADPHASE)
  assert collision_overflow(data) == expected


def test_empty_trajectory_rejected_before_capture():
  with pytest.raises(ValueError, match="nonempty"):
    measure(None, None, {"ctrl": []}, True, 20, {})
  with pytest.raises(ValueError, match="nonempty"):
    full_task_measure(None, None, {"ctrl": []}, 1, 256, 1024)

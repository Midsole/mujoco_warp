"""Check deterministic GPU collision output without changing the contact geometry."""

import dataclasses

import gripper_only
import mujoco
import numpy as np
import pytest
import warp as wp
from dense_sdf import attach_dense_sdf
from grasp_config import validate_config

import mujoco_warp as mjw


@pytest.mark.parametrize("sdf_mode", ["octree", "dense"])
@pytest.mark.parametrize("nworld", [1, 16, 512, 1024])
def test_frozen_gripper_contacts(sdf_mode, nworld):
  if not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  config = validate_config({"nworld": nworld, "sdf_initpoints": 10, "sdf_depth": 5})
  model = gripper_only.detach(config)
  initial = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, initial, 0)
  initial.qpos[int(model.joint("cat_free").qposadr[0]) + 2] -= 0.001
  mujoco.mj_forward(model, initial)
  with wp.ScopedDevice("cuda:0"):
    wm = mjw.put_model(model)
    if sdf_mode == "dense":
      attach_dense_sdf(model, wm, 17)
    original = mjw.put_data(model, initial, nworld=nworld, nconmax=128, njmax=256)
    mjw.collision(wm, original)
    reference_count = int(original.nacon.numpy()[0])
    assert reference_count > 0
    enabled = mjw.put_data(model, initial, nworld=nworld, nconmax=128, njmax=256)
    wm.opt.deterministic_contacts = True
    with wp.ScopedCapture() as capture:
      mjw.collision(wm, enabled)
    first = None
    for _ in range(10):
      wp.capture_launch(capture.graph)
      assert enabled.nacon.numpy()[0] == reference_count
      actual = {}
      for field in dataclasses.fields(enabled.contact):
        array = getattr(enabled.contact, field.name)
        if array.size:
          actual[field.name] = array.numpy()[:reference_count].copy()
      if first is None:
        first = actual
      for name, values in actual.items():
        assert values.tobytes() == first[name].tobytes(), name
      np.testing.assert_array_equal(actual["worldid"], np.sort(actual["worldid"]))
      # Each identical world produces the same ordered physical contact records.
      for world in range(1, nworld):
        for name, values in actual.items():
          if name != "worldid":
            assert values[actual["worldid"] == world].tobytes() == values[actual["worldid"] == 0].tobytes(), name

    # Compare the full contact multiset to the original path, independently of emission order.
    def rows(data):
      columns = []
      for name in sorted(first):
        columns.append(getattr(data.contact, name).numpy()[:reference_count].reshape(reference_count, -1))
      values = np.concatenate(columns, axis=1)
      return values[np.lexsort(tuple(values[:, i] for i in reversed(range(values.shape[1]))))]

    np.testing.assert_array_equal(rows(original), rows(enabled))

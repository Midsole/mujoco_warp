"""Independent checks for dense indexing, interpolation, and activation."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp
from dense_sdf import attach_dense_sdf
from dense_sdf import build_dense_sdf

import mujoco_warp as mjw
from mujoco_warp._src.collision_sdf import DenseSDF
from mujoco_warp._src.collision_sdf import VolumeData
from mujoco_warp._src.collision_sdf import attach_dense
from mujoco_warp._src.collision_sdf import sample_volume_grad
from mujoco_warp._src.collision_sdf import sample_volume_sdf


@wp.kernel
def _sample(volume: VolumeData, grids: DenseSDF, points: wp.array[wp.vec3], values_out: wp.array[wp.vec4]):
  i = wp.tid()
  cached = attach_dense(volume, grids, 0)
  gradient = sample_volume_grad(points[i], cached)
  values_out[i] = wp.vec4(sample_volume_sdf(points[i], cached), gradient[0], gradient[1], gradient[2])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_affine_distance_and_unnormalized_gradient(device):
  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  # Unequal dimensions, a nonzero offset, and nonsymmetric points catch indexing errors.
  dims = (3, 4, 5)
  lower = np.array([-1, -2, -3], dtype=np.float32)
  upper = -lower
  axes = [np.linspace(lo, hi, n) for lo, hi, n in zip(lower, upper, dims, strict=True)]
  xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
  gradient = np.array([2, -3, 4], dtype=np.float32)
  packed = np.zeros((7 + len(xyz), 4), dtype=np.float32)
  packed[7:, 0] = xyz @ gradient + 0.25
  packed[7:, 1:] = gradient
  points = np.array([[0.15, -0.6, 1.7], [-0.9, 1.8, -2.8], [0, 0, 0], [1, 2, 3]], dtype=np.float32)
  with wp.ScopedDevice(device):
    volume = VolumeData()
    assert not volume.dense_valid
    volume.valid = True
    volume.center = wp.vec3(0)
    volume.half_size = wp.vec3(*upper)
    grids = DenseSDF()
    grids.values = wp.array(packed, dtype=wp.vec4)
    grids.offsets = wp.array([7], dtype=int)
    grids.dims = wp.array([dims], dtype=wp.vec3i)
    grids.lower = wp.array([lower], dtype=wp.vec3)
    grids.inv_cell = wp.array([(np.array(dims) - 1) / (upper - lower)], dtype=wp.vec3)
    output = wp.empty(len(points), dtype=wp.vec4)
    wp.launch(_sample, dim=len(points), inputs=[volume, grids, wp.array(points, dtype=wp.vec3), output])
    actual = output.numpy()
  np.testing.assert_allclose(actual[:, 0], points @ gradient + 0.25, atol=2e-6)
  np.testing.assert_allclose(actual[:, 1:], np.tile(gradient, (len(points), 1)), atol=2e-6)


@pytest.mark.parametrize("resolution", [True, 2, 514, 4.5])
def test_invalid_resolution(resolution):
  with pytest.raises(ValueError, match="resolution"):
    build_dense_sdf(None, None, resolution)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cell_gradient_matches_trilinear_distance(device):
  from dense_sdf import _build_cell_gradients

  if device.startswith("cuda") and not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  n = 5
  lower = np.array([-1, -2, -3], dtype=np.float32)
  upper = -lower
  axes = [np.linspace(lo, hi, n) for lo, hi in zip(lower, upper, strict=True)]
  xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

  def polynomial(points):
    x, y, z = points.T
    distance = 0.25 + 2 * x - 3 * y + 4 * z + 5 * x * y + 6 * x * z + 7 * y * z + 8 * x * y * z
    gradient = np.stack([2 + 5 * y + 6 * z + 8 * y * z, -3 + 5 * x + 7 * z + 8 * x * z, 4 + 6 * x + 7 * y + 8 * x * y], axis=1)
    return distance, gradient

  packed = np.zeros((7 + len(xyz), 4), dtype=np.float32)
  packed[7:, 0] = polynomial(xyz)[0]
  points = np.array([[0.15, -0.6, 1.7], [-0.9, 1.8, -2.8], [0, 0, 0], [1, 2, 3]], dtype=np.float32)
  with wp.ScopedDevice(device):
    volume = VolumeData()
    volume.valid = True
    volume.center = wp.vec3(0)
    volume.half_size = wp.vec3(*upper)
    grids = DenseSDF()
    grids.values = wp.array(packed, dtype=wp.vec4)
    grids.offsets = wp.array([7], dtype=int)
    grids.dims = wp.array([[n, n, n]], dtype=wp.vec3i)
    grids.lower = wp.array([lower], dtype=wp.vec3)
    inv_cell = (n - 1) / (upper - lower)
    grids.inv_cell = wp.array([inv_cell], dtype=wp.vec3)
    grids.cell_gradient = True
    grids.gradient_offsets = wp.array([3], dtype=int)
    grids.cell_gradients = wp.empty(3 + 4 * (n - 1) ** 3, dtype=wp.vec3)
    wp.launch(_build_cell_gradients, dim=(n - 1) ** 3, inputs=[n, 7, 3, wp.vec3(*inv_cell), grids.values, grids.cell_gradients])
    output = wp.empty(len(points), dtype=wp.vec4)
    wp.launch(_sample, dim=len(points), inputs=[volume, grids, wp.array(points, dtype=wp.vec3), output])
    actual = output.numpy()
  distance, gradient = polynomial(points)
  np.testing.assert_allclose(actual[:, 0], distance, atol=2e-5)
  np.testing.assert_allclose(actual[:, 1:], gradient, atol=2e-5)


def test_dense_contacts_in_rotated_frame_and_disable():
  """Exercise baking and the real collision kernel, including cache disable/re-capture."""
  if not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  asset = Path(__file__).with_name("meshes") / "cube.obj"
  spec = mujoco.MjSpec.from_string(f'''<mujoco>
    <option sdf_initpoints="10" sdf_iterations="10"/>
    <asset><mesh name="cube" file="{asset}"/></asset>
    <worldbody>
      <body pos=".1 .2 .3" euler="20 30 40"><geom name="cube" type="sdf" mesh="cube"/></body>
      <body><freejoint/><geom type="sphere" size=".005" mass=".01"/></body>
    </worldbody>
  </mujoco>''')
  spec.meshes[0].octree_maxdepth = 4
  model = spec.compile()
  initial = mujoco.MjData(model)
  mujoco.mj_forward(model, initial)
  body = model.geom("cube").bodyid[0]
  initial.qpos[:3] = initial.xpos[body] + initial.xmat[body].reshape(3, 3) @ np.array([0.033, 0, 0.03])
  mujoco.mj_forward(model, initial)
  with wp.ScopedDevice("cuda:0"):
    warp_model = mjw.put_model(model)
    measurements = []
    for mode in ("octree", "dense", "disabled"):
      if mode == "dense":
        attach_dense_sdf(model, warp_model, 17)
      elif mode == "disabled":
        del warp_model.dense_sdf
      data = mjw.put_data(model, initial, nworld=2, nconmax=128, njmax=256)
      with wp.ScopedCapture() as capture:
        mjw.collision(warp_model, data)
      wp.capture_launch(capture.graph)
      count = int(data.nacon.numpy()[0])
      assert 0 < count < data.naconmax
      fields = [getattr(data.contact, name).numpy()[:count] for name in ("dist", "pos", "frame")]
      # Contact emission order across worlds is not deterministic.
      order = np.argsort(fields[0])
      measurements.append([value[order] for value in fields])
    for values in measurements[1:]:
      for actual, reference in zip(values, measurements[0], strict=True):
        np.testing.assert_allclose(actual, reference, atol=1e-4, rtol=1e-4)

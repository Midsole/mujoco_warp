"""Experimental dense distance/gradient cache for native mesh SDFs on Warp."""

import time

import mujoco
import numpy as np
import warp as wp

from mujoco_warp._src.collision_sdf import DenseSDF
from mujoco_warp._src.collision_sdf import VolumeData
from mujoco_warp._src.collision_sdf import sample_volume_grad
from mujoco_warp._src.collision_sdf import sample_volume_sdf


@wp.kernel
def _build_grid(
  # In:
  volume: VolumeData,
  lower: wp.vec3,
  cell: wp.vec3,
  resolution: int,
  offset: int,
  # Out:
  values_out: wp.array[wp.vec4],
):
  index = wp.tid()
  z = index % resolution
  y = (index // resolution) % resolution
  x = index // (resolution * resolution)
  point = lower + wp.cw_mul(wp.vec3(float(x), float(y), float(z)), cell)
  distance = sample_volume_sdf(point, volume)
  gradient = sample_volume_grad(point, volume)
  values_out[offset + index] = wp.vec4(distance, gradient[0], gradient[1], gradient[2])


@wp.kernel
def _build_cell_gradients(
  # In:
  resolution: int,
  offset: int,
  gradient_offset: int,
  inv_cell: wp.vec3,
  values: wp.array[wp.vec4],
  # Out:
  gradients_out: wp.array[wp.vec3],
):
  i = wp.tid()
  cells = resolution - 1
  z = i % cells
  y = (i // cells) % cells
  x = i // (cells * cells)
  b = offset + (x * resolution + y) * resolution + z
  sx, sy = resolution * resolution, resolution
  v000, v100 = values[b][0], values[b + sx][0]
  v010, v001 = values[b + sy][0], values[b + 1][0]
  v110, v101 = values[b + sx + sy][0], values[b + sx + 1][0]
  v011, v111 = values[b + sy + 1][0], values[b + sx + sy + 1][0]
  cxy = (v110 - v100) - (v010 - v000)
  cxz = (v101 - v100) - (v001 - v000)
  cyz = (v011 - v010) - (v001 - v000)
  cxyz = ((v111 - v110) - (v101 - v100)) - cyz
  out = gradient_offset + 4 * i
  gradients_out[out] = wp.cw_mul(wp.vec3(v100 - v000, v010 - v000, v001 - v000), inv_cell)
  gradients_out[out + 1] = wp.cw_mul(wp.vec3(cxy, cxy, cxz), inv_cell)
  gradients_out[out + 2] = wp.cw_mul(wp.vec3(cxz, cyz, cyz), inv_cell)
  gradients_out[out + 3] = wp.cw_mul(wp.vec3(cxyz), inv_cell)


def reference_volume(model, warp_model, mesh_id):
  """Return the octree view in the compiler's mesh coordinate frame."""
  root = int(model.mesh_octadr[mesh_id])
  if root < 0 or model.mesh_octnum[mesh_id] == 0:
    raise ValueError("Mesh has no native octree")
  volume = VolumeData()
  volume.center = wp.vec3(*model.oct_aabb[root, :3])
  volume.half_size = wp.vec3(*model.oct_aabb[root, 3:])
  volume.oct_aabb = warp_model.oct_aabb
  volume.oct_child = warp_model.oct_child
  volume.oct_coeff = warp_model.oct_coeff
  volume.root = root
  volume.valid = True
  return volume


def build_dense_sdf(model, warp_model, resolution=257, *, gradient_mode="cell"):
  """Bake the compiled octree into dense float32 (distance, gx, gy, gz) grids.

  This is a sampled approximation, not a mesh-to-exact-distance reconstruction.
  Cell mode caches derivatives of the distance interpolant; vertex mode blends sampled gradients.
  All worlds share the same buffers; building must happen before CUDA graph capture.
  """
  if isinstance(resolution, bool) or not isinstance(resolution, int) or not 3 <= resolution <= 513:
    raise ValueError("resolution must be an integer in [3, 513]")
  if gradient_mode not in ("vertex", "cell"):
    raise ValueError("gradient_mode must be 'vertex' or 'cell'")
  mesh_ids = sorted(
    set(int(model.geom_dataid[i]) for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SDF)
  )
  if any(model.geom_plugin[i] != -1 for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SDF):
    raise ValueError("Dense baking currently supports only native mesh SDFs")
  started = time.perf_counter()
  counts = resolution**3
  offsets = np.full(model.nmesh, -1, dtype=np.int32)
  gradient_offsets = np.full(model.nmesh, -1, dtype=np.int32)
  dims = np.zeros((model.nmesh, 3), dtype=np.int32)
  lower = np.zeros((model.nmesh, 3), dtype=np.float32)
  inv_cell = np.zeros((model.nmesh, 3), dtype=np.float32)
  grids = DenseSDF()
  with wp.ScopedDevice(warp_model.oct_coeff.device):
    grids.values = wp.empty(counts * len(mesh_ids), dtype=wp.vec4)
    grids.cell_gradient = gradient_mode == "cell"
    gradient_count = 4 * (resolution - 1) ** 3
    if grids.cell_gradient:
      grids.cell_gradients = wp.empty(gradient_count * len(mesh_ids), dtype=wp.vec3)
    metadata = []
    for index, mesh_id in enumerate(mesh_ids):
      volume = reference_volume(model, warp_model, mesh_id)
      center = np.array(volume.center)
      half_size = np.array(volume.half_size)
      cell = 2 * half_size / (resolution - 1)
      offsets[mesh_id] = index * counts
      dims[mesh_id] = resolution
      lower[mesh_id] = center - half_size
      inv_cell[mesh_id] = 1 / cell
      wp.launch(
        _build_grid,
        dim=counts,
        inputs=[volume, wp.vec3(*lower[mesh_id]), wp.vec3(*cell), resolution, int(offsets[mesh_id]), grids.values],
      )
      if grids.cell_gradient:
        gradient_offsets[mesh_id] = index * gradient_count
        wp.launch(
          _build_cell_gradients,
          dim=(resolution - 1) ** 3,
          inputs=[
            resolution,
            int(offsets[mesh_id]),
            int(gradient_offsets[mesh_id]),
            wp.vec3(*inv_cell[mesh_id]),
            grids.values,
            grids.cell_gradients,
          ],
        )
      metadata.append({"mesh": model.mesh(mesh_id).name, "resolution": resolution, "cell_m": cell.tolist()})
    grids.offsets = wp.array(offsets, dtype=int)
    if grids.cell_gradient:
      grids.gradient_offsets = wp.array(gradient_offsets, dtype=int)
    grids.dims = wp.array(dims, dtype=wp.vec3i)
    grids.lower = wp.array(lower, dtype=wp.vec3)
    grids.inv_cell = wp.array(inv_cell, dtype=wp.vec3)
    wp.synchronize()
  byte_count = counts * len(mesh_ids) * 16 + model.nmesh * (4 + 12 + 12 + 12)
  if grids.cell_gradient:
    byte_count += gradient_count * len(mesh_ids) * 12 + model.nmesh * 4
  return grids, {
    "meshes": metadata,
    "gradient_mode": gradient_mode,
    "grid_bytes": byte_count,
    "build_seconds": time.perf_counter() - started,
  }


def attach_dense_sdf(model, warp_model, resolution=257, *, gradient_mode="cell"):
  """Enable dense collision queries on this Warp model; return allocation/build metadata."""
  grids, metadata = build_dense_sdf(model, warp_model, resolution, gradient_mode=gradient_mode)
  warp_model.dense_sdf = grids
  return metadata

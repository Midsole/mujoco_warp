"""GPU rollout and contact traces for the SDF grasp example."""

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw


@wp.kernel
def _control(
  # In:
  targets: wp.array2d[float],
  index: wp.array[int],
  # Data out:
  ctrl_out: wp.array2d[float],
):
  world, joint = wp.tid()
  ctrl_out[world, joint] = targets[index[0], joint]


@wp.kernel
def _trace(
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  overflow_in: wp.array[int],
  # In:
  index: wp.array[int],
  # Out:
  positions_out: wp.array3d[float],
  velocities_out: wp.array3d[float],
  forces_out: wp.array3d[float],
  contacts_out: wp.array3d[int],
  warnings_out: wp.array[int],
):
  world = wp.tid()
  step = index[0]
  for j in range(qpos_in.shape[1]):
    positions_out[world, step, j] = qpos_in[world, j]
  for j in range(qvel_in.shape[1]):
    velocities_out[world, step, j] = qvel_in[world, j]
  for j in range(actuator_force_in.shape[1]):
    forces_out[world, step, j] = actuator_force_in[world, j]
  for j in range(4):
    contacts_out[world, step, j] = 0
  warnings_out[world] = warnings_out[world] | overflow_in[world]


@wp.kernel
def _trace_contacts(
  # Model:
  geom_bodyid: wp.array[int],
  # Data in:
  nacon_in: wp.array[int],
  # In:
  pairs: wp.array[wp.vec2i],
  worldids: wp.array[int],
  index: wp.array[int],
  cat: int,
  table: int,
  left: int,
  right: int,
  # Out:
  contacts_out: wp.array3d[int],
):
  c = wp.tid()
  if c >= nacon_in[0]:
    return
  world = worldids[c]
  a, b = pairs[c][0], pairs[c][1]
  group = int(-1)
  if a == cat or b == cat:
    other = a
    if a == cat:
      other = b
    group = 3
    if other == table:
      group = 2
    elif geom_bodyid[other] == left:
      group = 0
    elif geom_bodyid[other] == right:
      group = 1
  elif a == table or b == table:
    group = 3
  if group >= 0:
    wp.atomic_add(contacts_out, world, index[0], group, 1)


@wp.kernel
def _advance(
  # Out:
  index_out: wp.array[int],
):
  index_out[0] += 1


@wp.kernel
def _contact_force_trace(
  # Model:
  geom_bodyid: wp.array[int],
  # Data in:
  nacon_in: wp.array[int],
  # In:
  pairs: wp.array[wp.vec2i],
  worldids: wp.array[int],
  points: wp.array[wp.vec3],
  world_forces: wp.array[wp.spatial_vector],
  cat: int,
  table: int,
  left: int,
  right: int,
  sample_index: wp.array[int],
  # Out:
  positions_out: wp.array4d[float],
  forces_out: wp.array4d[float],
  groups_out: wp.array3d[int],
  totals_out: wp.array2d[int],
):
  c = wp.tid()
  if c >= nacon_in[0]:
    return
  a, b = pairs[c][0], pairs[c][1]
  if a != cat and b != cat:
    return
  world = worldids[c]
  sample = sample_index[0]
  kept = wp.atomic_add(totals_out, world, sample, 1)
  if kept >= positions_out.shape[2]:
    return
  other = a
  sign = 1.0
  if a == cat:
    other = b
    sign = -1.0
  group = 4
  if other == table:
    group = 1
  elif geom_bodyid[other] == left:
    group = 2
  elif geom_bodyid[other] == right:
    group = 3
  point = points[c]
  force = world_forces[c]
  for j in range(3):
    positions_out[world, sample, kept, j] = point[j]
    forces_out[world, sample, kept, j] = force[j] * sign
  groups_out[world, sample, kept] = group


@wp.kernel
def _contact_counts(
  # In:
  index: wp.array[int],
  totals: wp.array2d[int],
  capacity: int,
  # Out:
  counts_out: wp.array2d[int],
):
  world = wp.tid()
  counts_out[world, index[0]] = wp.min(totals[world, index[0]], capacity)


def rollout_warp(
  model, trajectory, *, nworld=1, nconmax=256, njmax=1024, progress=None, record_contact_forces=False, warp_model=None
):
  """Run independent worlds; batched states and forces have a leading world axis."""
  if isinstance(nworld, bool) or not isinstance(nworld, int) or nworld < 1:
    raise ValueError("nworld must be a positive integer")
  if not wp.is_cuda_available():
    raise RuntimeError("Warp grasp validation requires CUDA; use --engine=c for CPU")
  data = mujoco.MjData(model)
  data.qpos[:] = trajectory["qpos"][0]
  data.qvel[:] = trajectory["qvel"][0]
  mujoco.mj_forward(model, data)
  count = len(trajectory["ctrl"])
  with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(model) if warp_model is None else warp_model
    d = mjw.put_data(model, data, nworld=nworld, nconmax=nconmax, njmax=njmax)
    targets = wp.array(trajectory["ctrl"], dtype=float)
    index = wp.zeros(1, dtype=int)
    positions = wp.empty((nworld, count, model.nq))
    velocities = wp.empty((nworld, count, model.nv))
    forces = wp.empty((nworld, count, model.nu))
    contacts = wp.empty((nworld, count, 4), dtype=int)
    warnings = wp.zeros(nworld, dtype=int)
    ids = [model.geom("cat_sdf").id, model.geom("tabletop").id, model.body("gripper_link1").id, model.body("gripper_link2").id]
    with wp.ScopedCapture() as capture:
      wp.launch(_control, dim=(nworld, model.nu), inputs=[targets, index, d.ctrl])
      mjw.step(m, d)
      wp.launch(
        _trace,
        dim=nworld,
        inputs=[
          d.qpos,
          d.qvel,
          d.actuator_force,
          d.overflow,
          index,
          positions,
          velocities,
          forces,
          contacts,
          warnings,
        ],
      )
      wp.launch(
        _trace_contacts,
        dim=d.naconmax,
        inputs=[m.geom_bodyid, d.nacon, d.contact.geom, d.contact.worldid, index, *ids, contacts],
      )
      wp.launch(_advance, dim=1, inputs=[index])
    if record_contact_forces:
      sample_stride = max(1, round(0.01 / model.opt.timestep))
      sample_steps = np.arange(0, count, sample_stride, dtype=np.int32)
      # Bound sampled force storage for large batches; total contacts are still counted.
      capacity = min(nconmax, 256 if nworld == 1 else 64)
      sample_index = wp.zeros(1, dtype=int)
      contact_ids = wp.array(np.arange(d.naconmax, dtype=np.int32), dtype=int)
      world_forces = wp.zeros(d.naconmax, dtype=wp.spatial_vector)
      contact_positions = wp.zeros((nworld, len(sample_steps), capacity, 3))
      contact_forces = wp.zeros((nworld, len(sample_steps), capacity, 3))
      contact_groups = wp.zeros((nworld, len(sample_steps), capacity), dtype=int)
      contact_counts = wp.zeros((nworld, len(sample_steps)), dtype=int)
      contact_totals = wp.zeros((nworld, len(sample_steps)), dtype=int)
      with wp.ScopedCapture() as force_capture:
        mjw.contact_force(m, d, contact_ids, True, world_forces)
        wp.launch(
          _contact_force_trace,
          dim=d.naconmax,
          inputs=[
            m.geom_bodyid,
            d.nacon,
            d.contact.geom,
            d.contact.worldid,
            d.contact.pos,
            world_forces,
            *ids,
            sample_index,
            contact_positions,
            contact_forces,
            contact_groups,
            contact_totals,
          ],
        )
        wp.launch(_contact_counts, dim=nworld, inputs=[sample_index, contact_totals, capacity, contact_counts])
        wp.launch(_advance, dim=1, inputs=[sample_index])
    for step in range(count):
      wp.capture_launch(capture.graph)
      if record_contact_forces and step % sample_stride == 0:
        wp.capture_launch(force_capture.graph)
      if step % 2000 == 0:
        free = model.joint("cat_free").qposadr[0]
        q = d.qpos.numpy()[0]
        print(f"t={(step + 1) * model.opt.timestep:.1f}s object={q[free : free + 3].round(4)}", flush=True)
        if progress is not None:
          progress((step + 1) * model.opt.timestep)
    trace = {
      "qpos": positions.numpy(),
      "qvel": velocities.numpy(),
      "force": forces.numpy(),
      "contacts": contacts.numpy(),
      "warnings": warnings.numpy(),
    }
    if nworld == 1:
      for key in ("qpos", "qvel", "force", "contacts"):
        trace[key] = trace[key][0]
    if record_contact_forces:
      trace.update(
        contact_steps=sample_steps,
        contact_positions=contact_positions.numpy(),
        contact_forces=contact_forces.numpy(),
        contact_groups=contact_groups.numpy().astype(np.uint8),
        contact_counts=contact_counts.numpy(),
        contact_totals=contact_totals.numpy(),
      )
      if nworld == 1:
        for key in ("contact_positions", "contact_forces", "contact_groups", "contact_counts", "contact_totals"):
          trace[key] = trace[key][0]
    return trace

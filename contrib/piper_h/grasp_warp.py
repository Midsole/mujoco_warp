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
  joint = wp.tid()
  ctrl_out[0, joint] = targets[index[0], joint]


@wp.kernel
def _trace(
  # Model:
  geom_bodyid: wp.array[int],
  # Data in:
  qpos_in: wp.array2d[float],
  qvel_in: wp.array2d[float],
  actuator_force_in: wp.array2d[float],
  nacon_in: wp.array[int],
  overflow_in: wp.array[int],
  # In:
  pairs: wp.array[wp.vec2i],
  cat: int,
  table: int,
  left: int,
  right: int,
  # Out:
  index_out: wp.array[int],
  positions_out: wp.array2d[float],
  velocities_out: wp.array2d[float],
  forces_out: wp.array2d[float],
  contacts_out: wp.array2d[int],
  warnings_out: wp.array[int],
):
  step = index_out[0]
  for j in range(qpos_in.shape[1]):
    positions_out[step, j] = qpos_in[0, j]
  for j in range(qvel_in.shape[1]):
    velocities_out[step, j] = qvel_in[0, j]
  for j in range(actuator_force_in.shape[1]):
    forces_out[step, j] = actuator_force_in[0, j]
  for j in range(4):
    contacts_out[step, j] = 0
  for c in range(wp.min(nacon_in[0], pairs.shape[0])):
    a, b = pairs[c][0], pairs[c][1]
    if a == cat or b == cat:
      other = a
      if a == cat:
        other = b
      if other == table:
        contacts_out[step, 2] += 1
      elif geom_bodyid[other] == left:
        contacts_out[step, 0] += 1
      elif geom_bodyid[other] == right:
        contacts_out[step, 1] += 1
      else:
        contacts_out[step, 3] += 1
    elif a == table or b == table:
      contacts_out[step, 3] += 1
  warnings_out[0] = warnings_out[0] | overflow_in[0]
  index_out[0] += 1


@wp.kernel
def _contact_force_trace(
  # Model:
  geom_bodyid: wp.array[int],
  # Data in:
  nacon_in: wp.array[int],
  # In:
  pairs: wp.array[wp.vec2i],
  points: wp.array[wp.vec3],
  world_forces: wp.array[wp.spatial_vector],
  cat: int,
  table: int,
  left: int,
  right: int,
  # Out:
  sample_index_out: wp.array[int],
  positions_out: wp.array3d[float],
  forces_out: wp.array3d[float],
  groups_out: wp.array2d[int],
  counts_out: wp.array[int],
  totals_out: wp.array[int],
):
  sample = sample_index_out[0]
  kept = int(0)
  total = int(0)
  for c in range(wp.min(nacon_in[0], pairs.shape[0])):
    a, b = pairs[c][0], pairs[c][1]
    if a != cat and b != cat:
      continue
    total += 1
    if kept >= positions_out.shape[1]:
      continue
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
      positions_out[sample, kept, j] = point[j]
      forces_out[sample, kept, j] = force[j] * sign
    groups_out[sample, kept] = group
    kept += 1
  counts_out[sample] = kept
  totals_out[sample] = total
  sample_index_out[0] += 1


def rollout_warp(model, trajectory, *, nconmax=256, njmax=1024, progress=None, record_contact_forces=False):
  if not wp.is_cuda_available():
    raise RuntimeError("Warp grasp validation requires CUDA; use --engine=c for CPU")
  data = mujoco.MjData(model)
  data.qpos[:] = trajectory["qpos"][0]
  data.qvel[:] = trajectory["qvel"][0]
  mujoco.mj_forward(model, data)
  count = len(trajectory["ctrl"])
  with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(model)
    d = mjw.put_data(model, data, nworld=1, nconmax=nconmax, njmax=njmax)
    targets = wp.array(trajectory["ctrl"], dtype=float)
    index = wp.zeros(1, dtype=int)
    positions = wp.empty((count, model.nq))
    velocities = wp.empty((count, model.nv))
    forces = wp.empty((count, model.nu))
    contacts = wp.empty((count, 4), dtype=int)
    warnings = wp.zeros(1, dtype=int)
    ids = [model.geom("cat_sdf").id, model.geom("tabletop").id, model.body("gripper_link1").id, model.body("gripper_link2").id]
    with wp.ScopedCapture() as capture:
      wp.launch(_control, dim=model.nu, inputs=[targets, index, d.ctrl])
      mjw.step(m, d)
      wp.launch(
        _trace,
        dim=1,
        inputs=[
          m.geom_bodyid,
          d.qpos,
          d.qvel,
          d.actuator_force,
          d.nacon,
          d.overflow,
          d.contact.geom,
          *ids,
          index,
          positions,
          velocities,
          forces,
          contacts,
          warnings,
        ],
      )
    if record_contact_forces:
      sample_stride = max(1, round(0.01 / model.opt.timestep))
      sample_steps = np.arange(0, count, sample_stride, dtype=np.int32)
      capacity = min(nconmax, 256)
      sample_index = wp.zeros(1, dtype=int)
      contact_ids = wp.array(np.arange(nconmax, dtype=np.int32), dtype=int)
      world_forces = wp.zeros(nconmax, dtype=wp.spatial_vector)
      contact_positions = wp.zeros((len(sample_steps), capacity, 3))
      contact_forces = wp.zeros((len(sample_steps), capacity, 3))
      contact_groups = wp.zeros((len(sample_steps), capacity), dtype=int)
      contact_counts = wp.zeros(len(sample_steps), dtype=int)
      contact_totals = wp.zeros(len(sample_steps), dtype=int)
      with wp.ScopedCapture() as force_capture:
        mjw.contact_force(m, d, contact_ids, True, world_forces)
        wp.launch(
          _contact_force_trace,
          dim=1,
          inputs=[
            m.geom_bodyid,
            d.nacon,
            d.contact.geom,
            d.contact.pos,
            world_forces,
            *ids,
            sample_index,
            contact_positions,
            contact_forces,
            contact_groups,
            contact_counts,
            contact_totals,
          ],
        )
    for step in range(count):
      wp.capture_launch(capture.graph)
      if record_contact_forces and step % sample_stride == 0:
        wp.capture_launch(force_capture.graph)
      if step % 2000 == 0:
        free = model.joint("cat_free").qposadr[0]
        q = d.qpos.numpy()[0]
        print(f"t={(step + 1) * model.opt.timestep:.1f}s cat={q[free : free + 3].round(4)}", flush=True)
        if progress is not None:
          progress((step + 1) * model.opt.timestep)
    trace = {
      "qpos": positions.numpy(),
      "qvel": velocities.numpy(),
      "force": forces.numpy(),
      "contacts": contacts.numpy(),
      "warnings": warnings.numpy(),
    }
    if record_contact_forces:
      trace.update(
        contact_steps=sample_steps,
        contact_positions=contact_positions.numpy(),
        contact_forces=contact_forces.numpy(),
        contact_groups=contact_groups.numpy().astype(np.uint8),
        contact_counts=contact_counts.numpy(),
        contact_totals=contact_totals.numpy(),
      )
    return trace

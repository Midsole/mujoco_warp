"""GPU rollout and contact traces for the SDF grasp example."""

import mujoco
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


def rollout_warp(model, trajectory):
  if not wp.is_cuda_available():
    raise RuntimeError("Warp grasp validation requires CUDA; use --engine=c for CPU")
  data = mujoco.MjData(model)
  data.qpos[:] = trajectory["qpos"][0]
  data.qvel[:] = trajectory["qvel"][0]
  mujoco.mj_forward(model, data)
  count = len(trajectory["ctrl"])
  with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(model)
    d = mjw.put_data(model, data, nworld=1, nconmax=256, njmax=1024)
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
    for step in range(count):
      wp.capture_launch(capture.graph)
      if step % 2000 == 0:
        free = model.joint("cat_free").qposadr[0]
        q = d.qpos.numpy()[0]
        print(f"t={(step + 1) * model.opt.timestep:.1f}s cat={q[free : free + 3].round(4)}", flush=True)
    return {
      "qpos": positions.numpy(),
      "qvel": velocities.numpy(),
      "force": forces.numpy(),
      "contacts": contacts.numpy(),
      "warnings": warnings.numpy(),
    }

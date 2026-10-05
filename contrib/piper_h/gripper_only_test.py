"""Analytic and native inverse-dynamics checks for prescribed gripper motion."""

import grasp
import gripper_only as go
import mujoco
import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src import forward


@pytest.fixture(scope="module")
def models():
  config = {"sdf_depth": 5}
  full = grasp.build_model(config)
  reduced = go.detach(config)
  spec = go.detach(config, return_spec=True)
  for key in list(spec.keys):
    spec.delete(key)
  body = spec.body("gripper_base")
  body.mocap = False
  body.add_freejoint(name="test_free_root")
  return full, reduced, spec.compile()


def initial(model):
  d = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, d, 0)
  mujoco.mj_forward(model, d)
  return d


def test_reduced_model_and_control_mapping(models):
  full, reduced, _ = models
  a, b = initial(full), initial(reduced)
  assert reduced.nv == 8 and reduced.nu == 1 and reduced.nmocap == 1
  for name in ("gripper_base_collision", "gripper_link1_collision", "gripper_link2_collision", "cat_sdf"):
    np.testing.assert_allclose(a.geom_xpos[full.geom(name).id], b.geom_xpos[reduced.geom(name).id], atol=1e-6)
  traj = grasp.make_trajectory(full)
  mapped = go.reduced_trajectory(full, reduced, traj, np.zeros((1, len(traj["ctrl"]), 19)))
  np.testing.assert_array_equal(mapped["ctrl"][:, 0], traj["ctrl"][:, full.actuator("gripper_opening").id])


@pytest.mark.parametrize("invalid", ["shape", "nan", "quaternion"])
def test_reject_invalid_motion(models, invalid):
  _, reduced, _ = models
  data = initial(reduced)
  motion = np.zeros((1, 1, 19), np.float32)
  motion[0, 0, :7] = np.r_[data.mocap_pos[0], data.mocap_quat[0]]
  if invalid == "shape":
    motion = motion[0]
  elif invalid == "nan":
    motion[0, 0, 13] = np.nan
  else:
    motion[0, 0, 3:7] = 0
  with pytest.raises(ValueError, match="Root motion"):
    go.PrescribedGripper(reduced, motion)


@pytest.mark.parametrize(
  "velocity,acceleration",
  [
    ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    ([0.3, -0.2, 0.1, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
    ([0, 0, 0, 0.3, -0.4, 0.8], [0, 0, 0, 0, 0, 0]),
    ([0.2, 0.1, -0.3, 0.3, -0.4, 0.8], [2, -3, 4, -0.6, 0.7, -0.8]),
  ],
)
def test_bias_matches_free_root_inverse_dynamics(models, velocity, acceleration):
  _, reduced, free = models
  data = initial(reduced)
  data.qvel[reduced.joint("gripper_joint1").dofadr[0]] = 0.02
  data.qvel[reduced.joint("gripper_joint2").dofadr[0]] = -0.03
  motion = np.zeros((1, 1, 19), np.float32)
  motion[0, 0, :3], motion[0, 0, 3:7] = data.mocap_pos[0], data.mocap_quat[0]
  motion[0, 0, 7:13], motion[0, 0, 13:19] = velocity, acceleration
  ref = mujoco.MjData(free)
  root = free.joint("test_free_root")
  qa, da = int(root.qposadr[0]), int(root.dofadr[0])
  ref.qpos[qa : qa + 7] = motion[0, 0, :7]
  rotation = np.empty(9)
  mujoco.mju_quat2Mat(rotation, ref.qpos[qa + 3 : qa + 7])
  rotation = rotation.reshape(3, 3)
  ref.qvel[da : da + 3] = velocity[:3]
  ref.qvel[da + 3 : da + 6] = rotation.T @ velocity[3:]
  for name in ("gripper_joint1", "gripper_joint2", "cat_free"):
    n = 7 if name == "cat_free" else 1
    a, b = int(reduced.joint(name).qposadr[0]), int(free.joint(name).qposadr[0])
    ref.qpos[b : b + n] = data.qpos[a : a + n]
    a, b = int(reduced.joint(name).dofadr[0]), int(free.joint(name).dofadr[0])
    ref.qvel[b : b + min(n, 6)] = data.qvel[a : a + min(n, 6)]
  mujoco.mj_forward(free, ref)
  ref.qacc[:] = 0
  ref.qacc[da : da + 3] = acceleration[:3]
  ref.qacc[da + 3 : da + 6] = rotation.T @ acceleration[3:]
  expected = np.zeros(free.nv)
  mujoco.mj_rne(free, ref, 1, expected)
  with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(reduced)
    d = mjw.put_data(reduced, data, nconmax=256, njmax=1024)
    adapter = go.PrescribedGripper(reduced, motion)
    forward.fwd_position(m, d, factorize=False)
    forward.fwd_velocity(m, d)
    state = go.motion_state(m, d)
    wp.launch(
      go.moving_dynamics,
      dim=(1, reduced.nbody),
      inputs=[state, adapter.moving, adapter.root, adapter.motion, wp.zeros(1, dtype=int)],
    )
    bias = d.qfrc_bias.numpy()[0]
    cvel = d.cvel.numpy()[0]
  for name in ("gripper_joint1", "gripper_joint2"):
    np.testing.assert_allclose(bias[reduced.joint(name).dofadr[0]], expected[free.joint(name).dofadr[0]], atol=2e-6)
  # Compare physical body-origin velocities; subtree COM frames differ between models.
  for name in ("gripper_base", "gripper_link1", "gripper_link2"):
    body = reduced.body(name).id
    expected_v = np.zeros(6)
    mujoco.mj_objectVelocity(free, ref, mujoco.mjtObj.mjOBJ_BODY, free.body(name).id, expected_v, 0)
    origin = ref.xipos[free.body(name).id]
    actual_v = cvel[body].copy()
    actual_v[3:] += np.cross(actual_v[:3], origin - data.subtree_com[reduced.body_rootid[body]])
    np.testing.assert_allclose(actual_v, expected_v, atol=2e-6)


def test_export_matches_native_world_velocities(models):
  full, _, _ = models
  data = initial(full)
  positions = np.broadcast_to(data.qpos, (2, 8, full.nq)).copy()
  velocities = np.zeros((2, 8, full.nv))
  positions[1, :, 0] += 0.1
  velocities[1, :, :6] = [0.1, 0.2, -0.3, 0.2, 0.1, -0.2]
  motion = go.export_motion(full, positions, velocities)
  for world in range(2):
    data.qpos[:], data.qvel[:] = positions[world, 0], velocities[world, 0]
    mujoco.mj_forward(full, data)
    body = full.body("gripper_base").id
    jacp, jacr = np.zeros((3, full.nv)), np.zeros((3, full.nv))
    mujoco.mj_jacBody(full, data, jacp, jacr, body)
    np.testing.assert_allclose(motion[world, 1, :3], data.xpos[body], atol=2e-6)
    np.testing.assert_allclose(motion[world, 1, 7:13], np.r_[jacp @ data.qvel, jacr @ data.qvel], atol=2e-6)


def test_all_six_contact_rows_include_prescribed_motion(models):
  _, model, _ = models
  data = initial(model)
  motion = np.zeros((1, 1, 19), np.float32)
  motion[0, 0, :7] = np.r_[data.mocap_pos[0], data.mocap_quat[0]]
  motion[0, 0, 7:19] = [0.2, -0.3, 0.1, 0.4, 0.5, -0.6, 1, 2, 3, -0.2, 0.3, 0.4]
  r = np.array([0.1, 0.2, 0.3])
  with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(model)
    d = mjw.put_data(model, data, nconmax=1, njmax=32)
    d.nacon.fill_(1)
    d.contact.dim.fill_(6)
    d.contact.worldid.zero_()
    addresses = d.contact.efc_address.numpy()
    addresses[0, :6] = np.arange(6)
    wp.copy(d.contact.efc_address, wp.array(addresses, dtype=int))
    d.contact.geom.fill_(wp.vec2i(model.geom("cat_sdf").id, model.geom("gripper_link1_collision").id))
    d.contact.pos.fill_(wp.vec3(*(motion[0, 0, :3] + r)))
    d.contact.frame.fill_(wp.mat33(np.eye(3).ravel().tolist()))
    d.contact.solref.fill_(wp.vec2(0.004, 1))
    d.contact.solreffriction.zero_()
    from mujoco_warp._src.types import vec5

    d.contact.solimp.fill_(vec5(0.95, 0.99, 0.001, 0.5, 2))
    d.contact.dist.fill_(-0.001)
    d.contact.includemargin.zero_()
    d.efc.vel.zero_()
    d.efc.aref.zero_()
    adapter = go.PrescribedGripper(model, motion)
    state = go.motion_state(m, d)
    wp.launch(go.moving_contacts, dim=(1, 6), inputs=[state, adapter.moving, adapter.motion, wp.zeros(1, dtype=int)])
    vel, aref = d.efc.vel.numpy()[0, :6], d.efc.aref.numpy()[0, :6]
  v, w, a, alpha = (motion[0, 0, i : i + 3] for i in (7, 10, 13, 16))
  expected_v = np.r_[v + np.cross(w, r), w]
  expected_a = np.r_[a + np.cross(alpha, r) + np.cross(w, np.cross(w, r)), alpha]
  np.testing.assert_allclose(vel, expected_v, atol=2e-6)
  np.testing.assert_allclose(aref, -2 / (0.99 * 0.004) * expected_v - expected_a, atol=2e-4)


def test_stationary_adapter_matches_normal_step(models):
  _, model, _ = models
  data = initial(model)
  motion = np.zeros((1, 1, 19), np.float32)
  motion[0, 0, :7] = np.r_[data.mocap_pos[0], data.mocap_quat[0]]
  with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(model)
    a = mjw.put_data(model, data, nconmax=256, njmax=1024)
    b = mjw.put_data(model, data, nconmax=256, njmax=1024)
    adapter = go.PrescribedGripper(model, motion)
    adapter.state = go.motion_state(m, b)
    index = wp.zeros(1, dtype=int)
    with wp.ScopedCapture() as plain:
      mjw.step(m, a)
    with wp.ScopedCapture() as prescribed:
      adapter.step(m, b, index)
    for _ in range(20):
      wp.capture_launch(plain.graph)
      wp.capture_launch(prescribed.graph)
    np.testing.assert_allclose(a.qpos.numpy(), b.qpos.numpy(), atol=2e-6)
    np.testing.assert_allclose(a.qvel.numpy(), b.qvel.numpy(), atol=2e-5)


def test_fixed_trajectory_without_reference(models):
  from grasp_config import validate_config

  _, model, _ = models
  config = validate_config({"robot_mode": "gripper_only", "timestep": model.opt.timestep, "nworld": 16, "control_hz": 100})
  assert config["reference_run_id"] == ""
  trajectory = go.fixed_trajectory(model, config)
  motion = trajectory["root_motion"]
  assert motion.shape == (16, 12000, 19)
  np.testing.assert_array_equal(motion[0], motion[15])
  np.testing.assert_allclose(np.linalg.norm(motion[0, :, 3:7], axis=1), 1, atol=1e-6)
  # Check smooth Cartesian derivatives independently with centered differences.
  dt = model.opt.timestep
  np.testing.assert_allclose((motion[0, 2:, :3] - motion[0, :-2, :3]) / (2 * dt), motion[0, 1:-1, 7:10], atol=8e-5)
  interior = np.min(np.abs(trajectory["phase_times"][1:-1, None] - grasp.TIMES), axis=1) > 0.01
  difference = (motion[0, 2:, 7:10] - motion[0, :-2, 7:10]) / (2 * dt)
  np.testing.assert_allclose(difference[interior], motion[0, 1:-1, 13:16][interior], atol=2e-4)
  quaternion = motion[0, 1:-1, 3:7]
  omega = motion[0, 1:-1, 10:13]
  qdot = np.column_stack(
    (-0.5 * np.sum(omega * quaternion[:, 1:], axis=1), 0.5 * (quaternion[:, :1] * omega + np.cross(omega, quaternion[:, 1:])))
  )
  np.testing.assert_allclose((motion[0, 2:, 3:7] - motion[0, :-2, 3:7]) / (2 * dt), qdot, atol=1e-4)
  angular_difference = (motion[0, 2:, 10:13] - motion[0, :-2, 10:13]) / (2 * dt)
  np.testing.assert_allclose(angular_difference[interior], motion[0, 1:-1, 16:19][interior], atol=5e-4)
  np.testing.assert_allclose(motion[0, 0, 7:], 0, atol=1e-7)
  # Gripper targets honor the selected update stride without quantizing root motion.
  np.testing.assert_array_equal(trajectory["ctrl"].reshape(-1, 10), np.repeat(trajectory["ctrl"][::10], 10, axis=1))
  short = go.fixed_trajectory(model, {**config, "duration": 6})["root_motion"]
  np.testing.assert_allclose(short[:, :, :7], motion[:, ::2, :7], atol=1e-6)
  np.testing.assert_allclose(short[:, :, 7:13], 2 * motion[:, ::2, 7:13], atol=1e-6)
  np.testing.assert_allclose(short[:, :, 13:], 4 * motion[:, ::2, 13:], atol=1e-6)

  # Changing one scene must not change another scene, including before GPU upload.
  first = motion[0].copy()
  motion[15, :, 0] += 0.01
  np.testing.assert_array_equal(motion[0], first)
  np.testing.assert_allclose(motion[15, :, 0], first[:, 0] + 0.01, atol=1e-6)

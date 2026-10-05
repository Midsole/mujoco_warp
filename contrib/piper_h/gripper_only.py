"""Prescribed zero-DOF base for the PiPER's two dynamic prismatic fingers.

This opt-in step adapter is deliberately limited to the PiPER model: two sliding
fingers, elliptic contacts, no fluid forces, and implicitfast integration.
Motion columns are world position, wxyz quaternion, linear/angular velocity,
and linear/angular acceleration at the gripper_base origin (19 floats).
"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import warp as wp

from mujoco_warp._src import constraint
from mujoco_warp._src import forward
from mujoco_warp._src import solver
from mujoco_warp._src.types import vec5


def detach(config=None, *, object_path=None, return_spec=False):
  """Compile the configured full model with only the gripper tree and scene remaining."""
  import grasp

  full = grasp.build_model(config, object_path=object_path)
  initial = mujoco.MjData(full)
  mujoco.mj_resetDataKeyframe(full, initial, 0)
  mujoco.mj_forward(full, initial)
  base = full.body("gripper_base").id
  root = ET.fromstring(grasp.build_spec(config, object_path=object_path).to_xml())
  root.find("compiler").set("meshdir", str(grasp.SCENE.parent / "meshes"))
  world = root.find("worldbody")
  parents = {child: parent for parent in root.iter() for child in parent}
  body = root.find(".//body[@name='gripper_base']")
  parents[body].remove(body)
  body.set("mocap", "true")
  body.set("pos", " ".join(map(str, initial.xpos[base])))
  body.set("quat", " ".join(map(str, initial.xquat[base])))
  site = full.site("tool_center").id
  root_rotation = initial.xmat[base].reshape(3, 3)
  site_offset = root_rotation.T @ (initial.site_xpos[site] - initial.xpos[base])
  site_rotation = root_rotation.T @ initial.site_xmat[site].reshape(3, 3)
  site_quat = np.empty(4)
  mujoco.mju_mat2Quat(site_quat, site_rotation.ravel())
  ET.SubElement(
    body,
    "site",
    name="tool_center",
    pos=" ".join(map(str, site_offset)),
    quat=" ".join(map(str, site_quat)),
    size="0.004",
    rgba="0.1 0.85 0.9 1",
  )
  world.remove(world.find("body[@name='base_link']"))
  world.append(body)
  root.remove(root.find("keyframe"))
  root.find("size").attrib.pop("nkey", None)
  for actuator in list(root.find("actuator")):
    if actuator.get("name") != "gripper_opening":
      root.find("actuator").remove(actuator)
  names = {b.get("name") for b in world.iter("body")}
  for item in list(root.find("contact")):
    if item.get("body1") not in names or item.get("body2") not in names:
      root.find("contact").remove(item)
  used_meshes = {g.get("mesh") for g in root.iter("geom")}
  for mesh in list(root.find("asset")):
    if mesh.tag == "mesh" and mesh.get("name") not in used_meshes:
      root.find("asset").remove(mesh)
  spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
  model = spec.compile()
  qpos = model.qpos0.copy()
  for name in ("gripper_joint1", "gripper_joint2", "cat_free"):
    count = 7 if name == "cat_free" else 1
    a, b = int(full.joint(name).qposadr[0]), int(model.joint(name).qposadr[0])
    qpos[b : b + count] = initial.qpos[a : a + count]
  spec.add_key(name="home", qpos=qpos, ctrl=[initial.ctrl[full.actuator("gripper_opening").id]])
  model = spec.compile()
  assert model.nv == 8 and model.nu == 1 and model.nmocap == 1
  return spec if return_spec else model


@wp.struct
class MotionState:
  mocap_pos: wp.array2d[wp.vec3]
  mocap_quat: wp.array2d[wp.quat]
  nacon: wp.array[int]
  geom_bodyid: wp.array[int]
  body_dofnum: wp.array[int]
  body_dofadr: wp.array[int]
  body_jntadr: wp.array[int]
  body_rootid: wp.array[int]
  body_mass: wp.array2d[float]
  xaxis: wp.array2d[wp.vec3]
  xipos: wp.array2d[wp.vec3]
  subtree_com: wp.array2d[wp.vec3]
  qvel: wp.array2d[float]
  cvel: wp.array2d[wp.spatial_vector]
  qfrc_bias: wp.array2d[float]
  contact_dim: wp.array[int]
  contact_efc_address: wp.array2d[int]
  contact_worldid: wp.array[int]
  contact_geom: wp.array[wp.vec2i]
  contact_pos: wp.array[wp.vec3]
  contact_frame: wp.array[wp.mat33]
  contact_solref: wp.array[wp.vec2]
  contact_solreffriction: wp.array[wp.vec2]
  contact_solimp: wp.array[vec5]
  contact_dist: wp.array[float]
  contact_includemargin: wp.array[float]
  efc_vel: wp.array2d[float]
  efc_aref: wp.array2d[float]
  disableflags: int
  timestep: float


def motion_state(m, d):
  s = MotionState()
  s.mocap_pos = d.mocap_pos
  s.mocap_quat = d.mocap_quat
  s.nacon = d.nacon
  s.geom_bodyid = m.geom_bodyid
  s.body_dofnum = m.body_dofnum
  s.body_dofadr = m.body_dofadr
  s.body_jntadr = m.body_jntadr
  s.body_rootid = m.body_rootid
  s.body_mass = m.body_mass
  s.xaxis = d.xaxis
  s.xipos = d.xipos
  s.subtree_com = d.subtree_com
  s.qvel = d.qvel
  s.cvel = d.cvel
  s.qfrc_bias = d.qfrc_bias
  s.contact_dim = d.contact.dim
  s.contact_efc_address = d.contact.efc_address
  s.contact_worldid = d.contact.worldid
  s.contact_geom = d.contact.geom
  s.contact_pos = d.contact.pos
  s.contact_frame = d.contact.frame
  s.contact_solref = d.contact.solref
  s.contact_solreffriction = d.contact.solreffriction
  s.contact_solimp = d.contact.solimp
  s.contact_dist = d.contact.dist
  s.contact_includemargin = d.contact.includemargin
  s.efc_vel = d.efc.vel
  s.efc_aref = d.efc.aref
  s.disableflags = int(m.opt.disableflags)
  s.timestep = float(m.opt.timestep.numpy()[0])
  return s


@wp.kernel
def set_pose(motion: wp.array3d[float], index: wp.array[int], s: MotionState):
  world = wp.tid()
  k = index[0]
  s.mocap_pos[world, 0] = wp.vec3(motion[world, k, 0], motion[world, k, 1], motion[world, k, 2])
  s.mocap_quat[world, 0] = wp.quat(motion[world, k, 3], motion[world, k, 4], motion[world, k, 5], motion[world, k, 6])


@wp.func
def vector(motion: wp.array3d[float], world: int, k: int, offset: int):
  return wp.vec3(motion[world, k, offset], motion[world, k, offset + 1], motion[world, k, offset + 2])


@wp.kernel
def moving_contacts(s: MotionState, moving: wp.array[int], motion: wp.array3d[float], index: wp.array[int]):
  con, dim = wp.tid()
  if con >= s.nacon[0] or dim >= s.contact_dim[con]:
    return
  row = s.contact_efc_address[con, dim]
  if row < 0:
    return
  world = s.contact_worldid[con]
  k = index[0]
  b1 = s.geom_bodyid[s.contact_geom[con][0]]
  b2 = s.geom_bodyid[s.contact_geom[con][1]]
  sign = float(moving[b2] - moving[b1])
  if sign == 0.0:
    return
  omega = vector(motion, world, k, 10)
  alpha = vector(motion, world, k, 16)
  r = s.contact_pos[con] - vector(motion, world, k, 0)
  velocity = vector(motion, world, k, 7) + wp.cross(omega, r)
  acceleration = vector(motion, world, k, 13) + wp.cross(alpha, r) + wp.cross(omega, wp.cross(omega, r))
  # The moving base transports each slider's relative velocity as well.
  body = b1
  if moving[b2] != 0:
    body = b2
  if s.body_dofnum[body] == 1:
    joint = s.body_jntadr[body]
    rel = s.xaxis[world, joint] * s.qvel[world, s.body_dofadr[body]]
    acceleration += 2.0 * wp.cross(omega, rel)
  axis = s.contact_frame[con][dim % 3]
  dv = sign * wp.dot(axis, velocity)
  da = sign * wp.dot(axis, acceleration)
  if dim >= 3:
    dv = sign * wp.dot(axis, omega)
    da = sign * wp.dot(axis, alpha)
  ref = s.contact_solref[con]
  friction_ref = s.contact_solreffriction[con]
  if dim > 0 and (friction_ref[0] != 0.0 or friction_ref[1] != 0.0):
    ref = friction_ref
  kbi = constraint._contact_kbimp(
    s.disableflags,
    s.timestep,
    ref,
    s.contact_solimp[con],
    s.contact_dist[con] - s.contact_includemargin[con],
  )
  s.efc_vel[world, row] += dv
  s.efc_aref[world, row] -= kbi[1] * dv + da


@wp.kernel
def moving_dynamics(s: MotionState, moving: wp.array[int], root: int, motion: wp.array3d[float], index: wp.array[int]):
  world, body = wp.tid()
  if moving[body] == 0:
    return
  k = index[0]
  p = vector(motion, world, k, 0)
  omega = vector(motion, world, k, 10)
  alpha = vector(motion, world, k, 16)
  linear = vector(motion, world, k, 7)
  # cvel is represented at the tree COM, not the body's local origin.
  vcom = linear + wp.cross(omega, s.subtree_com[world, s.body_rootid[body]] - p)
  s.cvel[world, body] += wp.spatial_vector(omega, vcom)
  if body == root:
    return
  joint = s.body_jntadr[body]
  dof = s.body_dofadr[body]
  axis = s.xaxis[world, joint]
  r = s.xipos[world, body] - p
  acceleration = vector(motion, world, k, 13) + wp.cross(alpha, r) + wp.cross(omega, wp.cross(omega, r))
  acceleration += 2.0 * wp.cross(omega, axis * s.qvel[world, dof])
  s.qfrc_bias[world, dof] += s.body_mass[0, body] * wp.dot(axis, acceleration)


class PrescribedGripper:
  """Explicit opt-in adapter; the regular mjw.step path is untouched."""

  def __init__(self, model, motion):
    if model.nv != 8 or model.nu != 1 or model.nmocap != 1:
      raise ValueError("Prescribed gripper requires the reduced PiPER model")
    if model.opt.cone != mujoco.mjtCone.mjCONE_ELLIPTIC or model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
      raise ValueError("Prescribed gripper requires elliptic contacts and implicitfast")
    if model.opt.density or model.opt.viscosity or model.nflex:
      raise ValueError("Prescribed gripper does not support fluid forces or flex bodies")
    root = model.body("gripper_base").id
    for name in ("gripper_link1", "gripper_link2"):
      body = model.body(name).id
      joint = int(model.body_jntadr[body])
      if (
        model.body_parentid[body] != root
        or model.body_dofnum[body] != 1
        or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_SLIDE
      ):
        raise ValueError("Prescribed gripper requires two direct prismatic children")
    if motion.ndim != 3 or motion.shape[2] != 19 or not motion.shape[0] or not motion.shape[1]:
      raise ValueError("Root motion must have shape (worlds, steps, 19)")
    if not np.isfinite(motion).all() or not np.allclose(np.linalg.norm(motion[:, :, 3:7], axis=2), 1, atol=1e-4):
      raise ValueError("Root motion must be finite with unit wxyz quaternions")
    self.motion = wp.array(np.ascontiguousarray(motion, dtype=np.float32), dtype=float)
    flags = np.zeros(model.nbody, dtype=np.int32)
    self.root = model.body("gripper_base").id
    for name in ("gripper_base", "gripper_link1", "gripper_link2"):
      flags[model.body(name).id] = 1
    self.moving = wp.array(flags, dtype=int)
    self.state = None

  def step(self, m, d, index):
    if self.state is None:
      self.state = motion_state(m, d)
    wp.launch(set_pose, dim=d.nworld, inputs=[self.motion, index, self.state])
    forward.fwd_position(m, d, factorize=False)
    wp.launch(moving_contacts, dim=(d.naconmax, 6), inputs=[self.state, self.moving, self.motion, index])
    forward.fwd_velocity(m, d)
    wp.launch(moving_dynamics, dim=(d.nworld, m.nbody), inputs=[self.state, self.moving, self.root, self.motion, index])
    forward.fwd_actuation(m, d)
    forward.fwd_acceleration(m, d, factorize=True)
    solver.solve(m, d)
    forward.implicit(m, d)


@wp.kernel
def _load_reference(
  # In:
  positions: wp.array3d[float],
  velocities: wp.array3d[float],
  initial: wp.array[float],
  index: wp.array[int],
  # Data out:
  qpos_out: wp.array2d[float],
  qvel_out: wp.array2d[float],
):
  world, j = wp.tid()
  k = index[0]
  value = initial[j]
  if k > 0:
    value = positions[world, k - 1, j]
  qpos_out[world, j] = value
  if j < qvel_out.shape[1]:
    velocity = float(0.0)
    if k > 0:
      velocity = velocities[world, k - 1, j]
    qvel_out[world, j] = velocity


@wp.kernel
def _export_root(
  # Data in:
  xpos_in: wp.array2d[wp.vec3],
  xquat_in: wp.array2d[wp.quat],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  root: int,
  tree: int,
  index: wp.array[int],
  # Out:
  motion_out: wp.array3d[float],
):
  world = wp.tid()
  k = index[0]
  angular = wp.spatial_top(cvel_in[world, root])
  linear = wp.spatial_bottom(cvel_in[world, root]) + wp.cross(angular, xpos_in[world, root] - subtree_com_in[world, tree])
  for j in range(3):
    motion_out[world, k, j] = xpos_in[world, root][j]
    motion_out[world, k, j + 7] = linear[j]
    motion_out[world, k, j + 10] = angular[j]
  for j in range(4):
    motion_out[world, k, j + 3] = xquat_in[world, root][j]


def export_motion(full, positions, velocities):
  """Export pre-step root poses and analytic velocities from complete per-world states.

  Accelerations use forward differences of analytic spatial velocities, representing
  the actual velocity increment of the recorded discrete integration interval.
  """
  from parallel_efficiency_benchmark import _advance

  import mujoco_warp as mjw
  from mujoco_warp._src import smooth

  if positions.ndim == 2:
    positions, velocities = positions[None], velocities[None]
  nworld, count, _ = positions.shape
  with wp.ScopedDevice("cuda:0"):
    m = mjw.put_model(full)
    data = mujoco.MjData(full)
    mujoco.mj_resetDataKeyframe(full, data, 0)
    d = mjw.put_data(full, data, nworld=nworld, nconmax=1, njmax=1)
    q = wp.array(positions.astype(np.float32), dtype=float)
    v = wp.array(velocities.astype(np.float32), dtype=float)
    initial = wp.array(data.qpos.astype(np.float32), dtype=float)
    index = wp.zeros(1, dtype=int)
    result = wp.zeros((nworld, count, 19), dtype=float)
    root = full.body("gripper_base").id
    with wp.ScopedCapture() as capture:
      wp.launch(_load_reference, dim=(nworld, full.nq), inputs=[q, v, initial, index, d.qpos, d.qvel])
      smooth.kinematics(m, d)
      smooth.com_pos(m, d)
      smooth.com_vel(m, d)
      wp.launch(
        _export_root,
        dim=nworld,
        inputs=[d.xpos, d.xquat, d.subtree_com, d.cvel, root, int(full.body_rootid[root]), index, result],
      )
      wp.launch(_advance, dim=1, inputs=[index])
    for _ in range(count):
      wp.capture_launch(capture.graph)
    motion = result.numpy()
  if count > 1:
    motion[:, :-1, 13:19] = np.diff(motion[:, :, 7:13].astype(np.float64), axis=1) / full.opt.timestep
    motion[:, -1, 13:19] = motion[:, -2, 13:19]
  return motion


def reduced_trajectory(full, reduced, trajectory, motion):
  """Map controls by name; never reuse the first arm actuator as the gripper target."""
  result = dict(trajectory)
  result["ctrl"] = trajectory["ctrl"][:, [full.actuator("gripper_opening").id]].copy()
  result["qpos"] = reduced.key_qpos[:1].copy()
  result["qvel"] = np.zeros((1, reduced.nv))
  result["root_motion"] = motion
  return result


def fixed_trajectory(model, config):
  """Drive the root through fixed Cartesian waypoints with analytic derivatives."""
  import grasp
  from grasp_config import timing_settings

  settings = timing_settings(config)
  count = settings["physics_steps"]
  times = np.arange(count) * model.opt.timestep
  phase_times = times * grasp.TIMES[-1] / settings["duration"]
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  root = model.body("gripper_base").id
  site = model.site("tool_center").id
  local_offset = data.xmat[root].reshape(3, 3).T @ (data.site_xpos[site] - data.xpos[root])
  site_rotation = data.xmat[root].reshape(3, 3).T @ data.site_xmat[site].reshape(3, 3)
  rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]]) @ site_rotation.T
  down = np.empty(4)
  mujoco.mju_mat2Quat(down, rotation.ravel())
  bounds = grasp.object_bounds(model)
  width, _, height = bounds[1] - bounds[0]
  above_z = max(0.90, grasp.TABLE_HEIGHT + height + 0.07)
  grasp_z = max(grasp.TABLE_HEIGHT + 0.016, grasp.TABLE_HEIGHT + height / 2 - 0.005)
  lift_z = grasp_z + 0.11
  tcp = np.array(
    [
      (*grasp.PICK, above_z),
      (*grasp.PICK, grasp_z),
      (*grasp.PICK, lift_z),
      (*grasp.PLACE, lift_z),
      (*grasp.PLACE, grasp_z),
      (*grasp.PLACE, above_z),
    ]
  )
  above, pick, lift, transfer, place, retreat = tcp - rotation @ local_offset
  home = data.xpos[root]
  positions = np.array([home, home, above, pick, pick, lift, transfer, place, place, place, retreat, retreat])
  quaternions = np.tile(down, (len(grasp.TIMES), 1))
  quaternions[:2] = data.xquat[root]
  for index in range(1, len(quaternions)):
    if np.dot(quaternions[index], quaternions[index - 1]) < 0:
      quaternions[index] *= -1
  segments = np.minimum(np.searchsorted(grasp.TIMES, phase_times, side="right") - 1, len(grasp.TIMES) - 2)
  intervals = np.diff(grasp.TIMES)[segments] * settings["duration"] / grasp.TIMES[-1]
  u = (phase_times - grasp.TIMES[segments]) / np.diff(grasp.TIMES)[segments]
  blend = u**3 * (10 - 15 * u + 6 * u**2)
  speed = 30 * u**2 * (1 - u) ** 2 / intervals
  acceleration = 60 * u * (1 - u) * (1 - 2 * u) / intervals**2
  delta = positions[segments + 1] - positions[segments]
  motion = np.zeros((count, 19), np.float32)
  motion[:, :3] = positions[segments] + blend[:, None] * delta
  motion[:, 7:10] = speed[:, None] * delta
  motion[:, 13:16] = acceleration[:, None] * delta
  for segment in np.unique(segments):
    selected = segments == segment
    q0, q1 = quaternions[segment], quaternions[segment + 1]
    difference = np.empty(4)
    mujoco.mju_mulQuat(difference, q1, q0 * [1, -1, -1, -1])
    if difference[0] < 0:
      difference *= -1
    angular = np.empty(3)
    mujoco.mju_quat2Vel(angular, difference, 1)
    angle = np.linalg.norm(angular)
    fractions = blend[selected]
    relative = np.zeros((len(fractions), 4))
    relative[:, 0] = np.cos(angle * fractions / 2)
    if angle > 1e-12:
      relative[:, 1:] = np.sin(angle * fractions[:, None] / 2) * angular / angle
    # Hamilton product relative * q0, with world-frame rotation increments.
    motion[selected, 3] = relative[:, 0] * q0[0] - relative[:, 1:] @ q0[1:]
    motion[selected, 4:7] = relative[:, :1] * q0[1:] + q0[0] * relative[:, 1:] + np.cross(relative[:, 1:], q0[1:])
    motion[selected, 10:13] = speed[selected, None] * angular
    motion[selected, 16:19] = acceleration[selected, None] * angular
  opening = min(0.1, max(0.03, width + 0.02))
  openings = np.array([opening, opening, opening, opening, 0, 0, 0, 0, 0, opening, opening, opening])
  held_indices = np.arange(count) // settings["control_steps"] * settings["control_steps"]
  controls = openings[segments] + blend * (openings[segments + 1] - openings[segments])
  return {
    "ctrl": controls[held_indices, None],
    "times": times,
    "phase_times": phase_times,
    "duration": settings["duration"],
    "control_steps": settings["control_steps"],
    "effective_control_hz": settings["effective_control_hz"],
    "qpos": model.key_qpos[:1].copy(),
    "qvel": np.zeros((1, model.nv)),
    # Keep independently writable trajectories for future per-world randomization.
    "root_motion": np.repeat(motion[None], config["nworld"], axis=0),
  }

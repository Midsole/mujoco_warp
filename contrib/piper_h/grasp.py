"""Pick and place a cube or cat mesh using native octree SDF contacts."""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from grasp_config import timing_settings
from grasp_config import validate_config

SCENE = Path(__file__).resolve().with_name("grasp_scene.xml")
PICK = np.array([-0.16, -0.075])
PLACE = np.array([-0.16, 0.075])
TABLE_HEIGHT = 0.75
TIMES = np.array([0, 1, 3, 6, 8, 11, 14, 17, 18, 19.5, 21.5, 24])
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
ARM_ACTUATORS = tuple(f"position{i}" for i in range(1, 7))


def build_model(config=None):
  # Native scripts retain their 1 ms baseline; the dashboard passes its full configuration.
  config = validate_config({"timestep": 0.001, **(config or {})})
  if config["object_shape"] == "cat" and not SCENE.with_name("meshes").joinpath("cat_phone_stand.obj").exists():
    raise FileNotFoundError("Run prepare_cat.py --source /path/to/cat_phone_stand.stl first")
  spec = mujoco.MjSpec.from_file(str(SCENE))
  if config["object_shape"] == "cat":
    object_mesh = spec.mesh("grasp_cube")
    object_mesh.name = "cat_phone_stand"
    object_mesh.file = "cat_phone_stand.obj"
    for name in ("cat_sdf", "cat_visual"):
      spec.geom(name).meshname = "cat_phone_stand"
  if config["finger_collision"] == "box":
    for name in ("gripper_link1", "gripper_link2"):
      spec.delete(spec.geom(f"{name}_collision"))
      for part, pos, size in (
        ("root", [0, -0.0125, 0.004], [0.012, 0.0125, 0.004]),
        ("tip", [0, -0.050, 0.012], [0.018, 0.025, 0.01]),
      ):
        spec.body(name).add_geom(
          name=f"{name}_{part}_collision",
          default=spec.find_default("collision"),
          type=mujoco.mjtGeom.mjGEOM_BOX,
          pos=pos,
          size=size,
        )
  spec.option.timestep = config["timestep"]
  spec.option.iterations = config["solver_iterations"]
  spec.option.sdf_initpoints = config["sdf_initpoints"]
  spec.option.sdf_iterations = config["sdf_iterations"]
  spec.nconmax = config["nconmax"]
  spec.njmax = config["njmax"]
  sdf_meshes = {geom.meshname for geom in spec.geoms if geom.type == mujoco.mjtGeom.mjGEOM_SDF}
  for mesh in spec.meshes:
    if mesh.name in sdf_meshes:
      mesh.octree_maxdepth = config["sdf_depth"]
  # Set this before compiling so CPU's ngravcomp and Warp agree.
  robot_names = {"base_link", "flange_link", "gripper_base", "gripper_link1", "gripper_link2"}
  robot_names.update(f"link{i}" for i in range(1, 7))
  for body in spec.bodies:
    if body.name in robot_names:
      body.gravcomp = config["gravcomp"]
  for geom in spec.geoms:
    geom.condim = config["condim"]
    if geom.name == "tabletop":
      geom.priority = 2
      geom.solref = [config["table_contact_time"], 1]
      geom.friction = [config["table_friction"], config["table_torsional_friction"], config["table_rolling_friction"]]
    elif geom.name == "cat_sdf":
      geom.mass = config["cat_mass"]
      geom.solref = [config["cat_contact_time"], 1]
      geom.friction = [config["cat_friction"], config["cat_torsional_friction"], config["cat_rolling_friction"]]
  for joint in spec.joints:
    if joint.name in ARM_JOINTS:
      joint.damping[0] = config["arm_damping"][ARM_JOINTS.index(joint.name)]
    elif joint.name in ("gripper_joint1", "gripper_joint2"):
      joint.damping[0] = config["gripper_damping"]
  for actuator in spec.actuators:
    if actuator.name in ARM_ACTUATORS:
      index = ARM_ACTUATORS.index(actuator.name)
      kp, kv, limit = (config[key][index] for key in ("arm_kp", "arm_kv", "arm_force_limit"))
    elif actuator.name == "gripper_opening":
      kp, kv, limit = (config[key] for key in ("gripper_kp", "gripper_kv", "gripper_force_limit"))
    else:
      continue
    actuator.gainprm[0] = kp
    actuator.biasprm[1:3] = [-kp, -kv]
    actuator.forcerange = [-limit, limit]
  # MuJoCo's process-wide mesh cache omits octree depth from its key.
  # Recompiling after a different depth must rebuild the SDF octree.
  mujoco.mj_clearCache(mujoco.mj_getCache())
  model = spec.compile()
  # The included robot keyframe predates the object's free joint.
  free = model.joint("cat_free").qposadr[0]
  model.key_qpos[:, free : free + 7] = model.qpos0[free : free + 7]
  return model


def inverse_kinematics(model, position, seed, rotation=None):
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  addresses = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
  dofs = np.array([model.joint(name).dofadr[0] for name in ARM_JOINTS])
  limits = np.array([model.joint(name).range for name in ARM_JOINTS])
  site = model.site("tool_center").id
  if rotation is None:
    rotation = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])
  desired_quat = np.empty(4)
  mujoco.mju_mat2Quat(desired_quat, rotation.ravel())
  q = np.array(seed, dtype=float)
  for _ in range(400):
    data.qpos[addresses] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    current_quat = np.empty(4)
    mujoco.mju_mat2Quat(current_quat, data.site_xmat[site])
    # Use a world-frame angular error to match the world-frame rotational Jacobian.
    conjugate = current_quat * np.array([1, -1, -1, -1])
    difference = np.empty(4)
    mujoco.mju_mulQuat(difference, desired_quat, conjugate)
    angular_error = np.empty(3)
    mujoco.mju_quat2Vel(angular_error, difference, 1)
    position_error = np.asarray(position) - data.site_xpos[site]
    if np.linalg.norm(position_error) < 1e-5 and np.linalg.norm(angular_error) < 1e-4:
      return q
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site)
    jac = np.vstack([jacp[:, dofs], 0.3 * jacr[:, dofs]])
    error = np.r_[position_error, 0.3 * angular_error]
    dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), error)
    q = np.clip(q + np.clip(dq, -0.15, 0.15), limits[:, 0], limits[:, 1])
  raise ValueError(f"Unreachable grasp waypoint: {position}")


def make_trajectory(model, config=None):
  """Build the native 24 s baseline, or rescale and hold targets for a configured run."""
  addresses = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
  actuators = np.array([model.actuator(name).id for name in ARM_ACTUATORS])
  gripper = model.actuator("gripper_opening").id
  home = model.key_qpos[0, addresses].copy()
  pick_tcp = PICK
  place_tcp = PLACE + [-0.0006, 0.0022]
  targets = [(*pick_tcp, 0.90), (*pick_tcp, 0.79), (*pick_tcp, 0.89), (*PLACE, 0.89), (*place_tcp, 0.7988), (*place_tcp, 0.90)]
  cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "grasp_cube") >= 0
  if cube:
    targets = [(*PICK, 0.90), (*PICK, 0.775), (*PICK, 0.885), (*PLACE, 0.885), (*PLACE, 0.775), (*PLACE, 0.90)]
  seed = np.array([0, 1.3, -1.3, 0, 1.2, 1.57])
  poses = []
  held_quat = np.empty(4)
  # Approximate the small tilt of the compliant grasp when approaching the table.
  # These are simulation waypoints, not a calibrated object pose controller.
  mujoco.mju_euler2Quat(held_quat, np.deg2rad([-9, 2, 0]), "XYZ")
  held_rotation = np.empty(9)
  mujoco.mju_quat2Mat(held_rotation, held_quat)
  drop_rotation = held_rotation.reshape(3, 3).T @ np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])
  for index, target in enumerate(targets):
    seed = inverse_kinematics(model, target, seed, rotation=drop_rotation if index >= 4 and not cube else None)
    poses.append(seed)
  above, grasp, lift, transfer, place, retreat = poses
  qkeys = np.array([home, home, above, grasp, grasp, lift, transfer, place, place, place, retreat, retreat])
  openings = [0.08, 0.08, 0.08, 0.08, 0, 0, 0, 0, 0, 0.08, 0.08, 0.08]
  keys = np.tile(model.key_ctrl[0], (len(TIMES), 1))
  keys[:, actuators] = qkeys
  keys[:, gripper] = openings
  if config is None:
    # Existing benchmarks and command-line playback retain the native baseline.
    times = np.arange(round(TIMES[-1] / model.opt.timestep) + 1) * model.opt.timestep
    phase_times, control_times = times, times
    settings = {"duration": float(TIMES[-1]), "control_steps": 1, "effective_control_hz": 1 / model.opt.timestep}
  else:
    config = validate_config(config)
    if not np.isclose(model.opt.timestep, config["timestep"]):
      raise ValueError("轨迹配置的物理步长必须与模型一致")
    settings = timing_settings(config)
    indices = np.arange(settings["physics_steps"])
    times = indices * model.opt.timestep
    phase_times = times * TIMES[-1] / settings["duration"]
    control_times = (indices // settings["control_steps"]) * settings["control_steps"] * model.opt.timestep
    control_times = control_times * TIMES[-1] / settings["duration"]
  segments = np.minimum(np.searchsorted(TIMES, control_times, side="right") - 1, len(TIMES) - 2)
  u = ((control_times - TIMES[segments]) / (TIMES[segments + 1] - TIMES[segments]))[:, None]
  blend = u**3 * (10 + u * (-15 + 6 * u))
  controls = keys[segments] + blend * (keys[segments + 1] - keys[segments])
  return {
    "ctrl": controls,
    "times": times,
    "phase_times": phase_times,
    "duration": settings["duration"],
    "control_steps": settings["control_steps"],
    "effective_control_hz": settings["effective_control_hz"],
    "qpos": model.key_qpos[:1].copy(),
    "qvel": np.zeros((1, model.nv)),
  }


def rollout_cpu(model, trajectory, progress=None, record_contact_forces=False):
  data = mujoco.MjData(model)
  data.qpos[:] = trajectory["qpos"][0]
  data.qvel[:] = trajectory["qvel"][0]
  mujoco.mj_forward(model, data)
  controls = trajectory["ctrl"]
  positions = np.empty((len(controls), model.nq))
  velocities = np.empty((len(controls), model.nv))
  forces = np.empty((len(controls), model.nu))
  contacts = np.zeros((len(controls), 4), dtype=int)
  cat = model.geom("cat_sdf").id
  finger_bodies = [model.body(name).id for name in ("gripper_link1", "gripper_link2")]
  table = model.geom("tabletop").id
  if record_contact_forces:
    sample_stride = max(1, round(0.01 / model.opt.timestep))
    sample_steps = np.arange(0, len(controls), sample_stride, dtype=np.int32)
    capacity = min(model.nconmax, 256)
    contact_positions = np.zeros((len(sample_steps), capacity, 3), dtype=np.float32)
    contact_forces = np.zeros_like(contact_positions)
    contact_groups = np.zeros((len(sample_steps), capacity), dtype=np.uint8)
    contact_counts = np.zeros(len(sample_steps), dtype=np.int32)
    contact_totals = np.zeros(len(sample_steps), dtype=np.int32)
  for step, ctrl in enumerate(controls):
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)
    positions[step], velocities[step], forces[step] = data.qpos, data.qvel, data.actuator_force
    for contact in data.contact:
      a, b = contact.geom
      if a == cat or b == cat:
        other = b if a == cat else a
        if other == table:
          contacts[step, 2] += 1
        elif model.geom_bodyid[other] in finger_bodies:
          contacts[step, finger_bodies.index(model.geom_bodyid[other])] += 1
        else:
          contacts[step, 3] += 1
      elif a == table or b == table:
        contacts[step, 3] += 1
    if record_contact_forces and step % sample_stride == 0:
      sample = step // sample_stride
      for contact_id, contact in enumerate(data.contact):
        a, b = contact.geom
        if a != cat and b != cat:
          continue
        contact_totals[sample] += 1
        slot = contact_counts[sample]
        if slot >= capacity:
          continue
        local_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, local_force)
        force_on_cat = contact.frame.reshape(3, 3).T @ local_force[:3]
        if a == cat:
          force_on_cat = -force_on_cat
        other = b if a == cat else a
        body = model.geom_bodyid[other]
        group = 1 if other == table else 2 if body == finger_bodies[0] else 3 if body == finger_bodies[1] else 4
        contact_positions[sample, slot] = contact.pos
        contact_forces[sample, slot] = force_on_cat
        contact_groups[sample, slot] = group
        contact_counts[sample] += 1
    if step % 2000 == 0:
      free = model.joint("cat_free").qposadr[0]
      print(f"t={data.time:.1f}s object={data.qpos[free : free + 3].round(4)} contacts={contacts[step]}", flush=True)
      if progress is not None:
        progress(float(data.time))
  trace = {"qpos": positions, "qvel": velocities, "force": forces, "contacts": contacts, "warnings": data.warning.number.copy()}
  if record_contact_forces:
    trace.update(
      contact_steps=sample_steps,
      contact_positions=contact_positions,
      contact_forces=contact_forces,
      contact_groups=contact_groups,
      contact_counts=contact_counts,
      contact_totals=contact_totals,
    )
  return trace


def validate(model, trajectory, trace, config=None):
  config = validate_config(config or {})
  times = trajectory.get("phase_times", trajectory["times"])
  free = model.joint("cat_free").qposadr[0]
  cat = trace["qpos"][:, free : free + 3]
  carried = (times >= 10.5) & (times <= 14)
  final = times >= 23
  contact = trace["contacts"]
  geom = model.geom("cat_sdf")
  mesh = model.mesh(int(geom.dataid[0]))
  vertices = model.mesh_vert[mesh.vertadr[0] : mesh.vertadr[0] + mesh.vertnum[0]]
  rotation = np.empty(9)
  mujoco.mju_quat2Mat(rotation, geom.quat)
  vertices = vertices @ rotation.reshape(3, 3).T + geom.pos
  bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
  quat = trace["qpos"][:, free + 3 : free + 7]
  w, x, y, z = quat.T
  rotation_z = np.column_stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
  bottom = cat[:, 2] + np.minimum(rotation_z * bounds[0], rotation_z * bounds[1]).sum(axis=1)
  joint_names = (*ARM_JOINTS, "gripper_joint1", "gripper_joint2")
  addresses = [model.joint(name).qposadr[0] for name in joint_names]
  limits = np.array([model.joint(name).range for name in joint_names])
  joint_positions = trace["qpos"][:, addresses]
  limit_error = np.maximum(limits[:, 0] - joint_positions, joint_positions - limits[:, 1])
  force_error = np.maximum(model.actuator_forcerange[:, 0] - trace["force"], trace["force"] - model.actuator_forcerange[:, 1])
  base = model.body("base_link")
  results = {
    "minimum_carried_height_m": float(np.min(bottom[carried] - TABLE_HEIGHT)),
    "placement_error_m": float(np.linalg.norm(cat[-1, :2] - PLACE)),
    "placement_tolerance_m": config["placement_tolerance"],
    "final_position_m": cat[-1].tolist(),
    "both_fingers_contact": bool(np.any(np.all(contact[(times >= 7) & (times <= 14), :2] > 0, axis=1))),
    "carried_contact_fraction": float(np.mean(np.all(contact[carried, :2] > 0, axis=1))),
    "unexpected_contacts": int(np.sum(contact[:, 3])),
    "final_speed": float(np.max(np.abs(trace["qvel"][final, model.joint("cat_free").dofadr[0] :]))),
    "final_drift_m": float(np.max(np.ptp(cat[final], axis=0))),
    "joint_limit_error": float(np.max(np.maximum(limit_error, 0))),
    "force_limit_error": float(np.max(np.maximum(force_error, 0))),
  }
  checks = [
    np.all(np.isfinite(trace["qpos"])),
    np.all(np.isfinite(trace["qvel"])),
    not np.any(trace["warnings"]),
    results["minimum_carried_height_m"] >= config["minimum_carried_height"],
    results["placement_error_m"] <= config["placement_tolerance"],
    results["both_fingers_contact"],
    results["carried_contact_fraction"] >= config["minimum_contact_fraction"],
    results["unexpected_contacts"] == 0,
    results["final_speed"] < 0.1,
    results["final_drift_m"] <= config["maximum_final_drift"],
    results["joint_limit_error"] < 0.001,
    results["force_limit_error"] < 1e-4,
    base.jntnum[0] == 0 and np.allclose(base.pos, [-0.5, 0, 0.75]),
    bool(np.any(contact[final, 2] > 0)),
  ]
  results["passed"] = bool(all(checks))
  return results


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--engine", choices=("warp", "c"), default="warp")
  parser.add_argument("--viewer", choices=("mujoco", "viser"), default="mujoco")
  parser.add_argument("--headless", action="store_true", help="run and validate a complete pick-and-place without a window")
  parser.add_argument("--finger-collision", choices=("sdf", "box"), default="sdf")
  parser.add_argument("--object-shape", choices=("cube", "cat"), default="cube")
  args = parser.parse_args()
  print("正在编译碰撞模型的 SDF 八叉树，请稍候……", flush=True)
  model = build_model({"finger_collision": args.finger_collision, "object_shape": args.object_shape})
  print("正在生成抓取轨迹……", flush=True)
  trajectory = make_trajectory(model)
  if args.headless:
    if args.engine == "warp":
      from grasp_warp import rollout_warp

      trace = rollout_warp(model, trajectory)
    else:
      trace = rollout_cpu(model, trajectory)
    result = validate(model, trajectory, trace)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1
  with tempfile.TemporaryDirectory(prefix="piper_h_grasp_") as directory:
    binary = Path(directory) / "scene.mjb"
    replay = Path(directory) / "trajectory.npz"
    mujoco.mj_saveModel(model, str(binary))
    np.savez(replay, **trajectory)
    return subprocess.call(
      [
        sys.executable,
        str(SCENE.with_name("grasp_viewer.py")),
        str(binary),
        f"--replay={replay}",
        f"--engine={args.engine}",
        f"--viewer={args.viewer}",
        "--nworld=1",
        "--noise_std=0",
        "--nconmax=256",
        "--njmax=1024",
      ]
    )


if __name__ == "__main__":
  try:
    sys.exit(main())
  except KeyboardInterrupt:
    sys.exit(130)

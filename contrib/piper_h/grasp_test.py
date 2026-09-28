"""Validate the full-resolution SDF and contact-driven transport on both backends."""

import hashlib
import json
from pathlib import Path

import grasp
import mujoco
import numpy as np
import prepare_cat
import pytest
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src.io import load_trajectory


@pytest.fixture(scope="module")
def model():
  return grasp.build_model()


def test_lossless_conversion(tmp_path):
  vertices = np.array([[0, 0, 0], [40, 0, 0], [0, 100, 0], [0, 0, 70]], dtype=np.float32)
  faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
  records = np.zeros(4, dtype=prepare_cat.STL_RECORD)
  records["vertices"] = vertices[faces]
  source = tmp_path / "source.stl"
  original = bytes(80) + (4).to_bytes(4, "little") + records.tobytes()
  source.write_bytes(original)
  metadata = prepare_cat.prepare(source, tmp_path)
  lines = (tmp_path / "meshes/cat_phone_stand.obj").read_text().splitlines()
  converted_vertices = np.array([line.split()[1:] for line in lines if line.startswith("v ")], dtype=float)
  converted_faces = np.array([line.split()[1:] for line in lines if line.startswith("f ")], dtype=int) - 1
  expected = (vertices[faces].astype(float) - metadata["source_offset_mm"]) * 0.001
  np.testing.assert_array_equal(converted_vertices[converted_faces], expected)
  assert source.read_bytes() == original
  assert metadata["triangles"] == 4 and metadata["nonmanifold_edges"] == 0


def test_asset_model_and_replay(tmp_path):
  model = grasp.build_model({"object_shape": "cat"})
  directory = grasp.SCENE.parent
  for record, urdf in (("provenance.json", "piper_h.urdf"), ("gripper_provenance.json", "piper_h_with_gripper.urdf")):
    provenance = json.loads((directory / record).read_text())
    for name, digest in provenance["files"].items():
      assert hashlib.sha256((directory / "meshes" / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((directory / urdf).read_bytes()).hexdigest() == provenance["urdf_sha256"]
  np.testing.assert_allclose(model.body("base_link").pos, [-0.5, 0, 0.75])
  np.testing.assert_allclose(model.geom("tabletop").size, [0.6, 0.4, 0.02])
  assert model.geom("tabletop").pos[2] + model.geom("tabletop").size[2] == pytest.approx(0.75)
  for name, actuator in zip(grasp.ARM_JOINTS, grasp.ARM_ACTUATORS, strict=True):
    np.testing.assert_allclose(model.actuator(actuator).ctrlrange, model.joint(name).range)
  np.testing.assert_allclose(model.actuator("gripper_opening").ctrlrange, [0, 0.1])
  metadata = json.loads((directory / "cat_provenance.json").read_text())
  obj = directory / "meshes/cat_phone_stand.obj"
  assert hashlib.sha256(obj.read_bytes()).hexdigest() == metadata["obj_sha256"]
  lines = obj.read_text().splitlines()
  vertices = np.array([line.split()[1:] for line in lines if line.startswith("v ")], dtype=float)
  faces = np.array([line.split()[1:] for line in lines if line.startswith("f ")], dtype=int) - 1
  assert faces.shape == (468114, 3)
  np.testing.assert_allclose([vertices.min(0), vertices.max(0)], metadata["bounds_m"], atol=1e-15)
  edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
  assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
  source = Path(metadata["source"])
  if source.exists():
    raw, original_vertices, original_faces = prepare_cat.read_stl(source)
    assert hashlib.sha256(raw).hexdigest() == metadata["source_sha256"]
    expected = (original_vertices[original_faces] - metadata["source_offset_mm"]) * metadata["scale"]
    np.testing.assert_array_equal(vertices[faces], expected)
  assert (model.nq, model.nv, model.nu) == (15, 14, 7)
  assert model.joint("cat_free").type == mujoco.mjtJoint.mjJNT_FREE
  assert model.geom("cat_sdf").type == mujoco.mjtGeom.mjGEOM_SDF
  assert model.geom("cat_visual").contype == 0 and model.geom("cat_visual").conaffinity == 0
  mesh = model.mesh("cat_phone_stand").id
  start, count = model.mesh_octadr[mesh], model.mesh_octnum[mesh]
  assert count > 0
  depths = model.oct_depth[start : start + count]
  assert depths.max() == 8
  assert model.body_gravcomp[model.body("cat_phone_stand").id] == 0
  assert model.body("cat_phone_stand").mass == pytest.approx(0.1)
  assert model.ngravcomp == 11
  assert model.neq == 1 and model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
  assert (model.opt.sdf_initpoints, model.opt.sdf_iterations) == (40, 10)
  np.testing.assert_allclose(model.actuator("gripper_opening").forcerange, [-10, 10])
  trajectory = grasp.make_trajectory(model)
  controls = trajectory["ctrl"]
  assert np.all(controls >= model.actuator_ctrlrange[:, 0])
  assert np.all(controls <= model.actuator_ctrlrange[:, 1])
  velocity = np.diff(controls, axis=0) / model.opt.timestep
  for time in grasp.TIMES[1:-1]:
    step = round(time / model.opt.timestep)
    assert np.max(np.abs(velocity[step - 1 : step + 1])) < 1e-4
  binary, replay = tmp_path / "scene.mjb", tmp_path / "trajectory.npz"
  mujoco.mj_saveModel(model, str(binary))
  np.savez(replay, **trajectory)
  reloaded = mujoco.MjModel.from_binary_path(str(binary))
  data = mujoco.MjData(reloaded)
  np.testing.assert_array_equal(load_trajectory(str(replay), reloaded, data), controls)
  free = reloaded.joint("cat_free").qposadr[0]
  np.testing.assert_allclose(data.qpos[free : free + 7], [-0.16, -0.075, 0.7505, 1, 0, 0, 0])


@pytest.fixture(scope="module")
def probe_model():
  asset = grasp.SCENE.parent / "meshes/cat_phone_stand.obj"
  spec = mujoco.MjSpec.from_string(f'''<mujoco>
    <option sdf_initpoints="40" sdf_iterations="10"/>
    <asset><mesh name="cat" file="{asset}"/></asset>
    <worldbody>
      <geom name="cat" type="sdf" mesh="cat"/>
      <body pos="1 0 0"><freejoint/><geom type="sphere" size=".002" mass=".001"/></body>
    </worldbody>
  </mujoco>''')
  spec.meshes[0].octree_maxdepth = 8
  return spec.compile()


def test_finger_collision_models_preserve_dynamics():
  sdf = grasp.build_model({"finger_collision": "sdf", "sdf_depth": 5})
  box = grasp.build_model({"finger_collision": "box", "sdf_depth": 5})
  for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "qpos0", "jnt_range", "actuator_gainprm"):
    np.testing.assert_array_equal(getattr(sdf, field), getattr(box, field))
  for name in ("gripper_link1", "gripper_link2"):
    for model, expected_count in ((sdf, 1), (box, 2)):
      body = model.body(name).id
      active = np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0))
      assert len(active) == expected_count
      assert model.geom(f"{name}_visual").contype == 0
    geom = sdf.geom(f"{name}_collision")
    mesh = sdf.mesh(f"{name}_mesh").id
    assert geom.type == mujoco.mjtGeom.mjGEOM_SDF and geom.dataid == mesh
    start, count = sdf.mesh_octadr[mesh], sdf.mesh_octnum[mesh]
    assert count > 0 and sdf.oct_depth[start : start + count].max() == 5
    np.testing.assert_allclose(geom.pos, sdf.geom(f"{name}_visual").pos)
    np.testing.assert_allclose(geom.quat, sdf.geom(f"{name}_visual").quat)
    for part, pos, size in (
      ("root", [0, -0.0125, 0.004], [0.012, 0.0125, 0.004]),
      ("tip", [0, -0.050, 0.012], [0.018, 0.025, 0.01]),
    ):
      geom = box.geom(f"{name}_{part}_collision")
      assert geom.type == mujoco.mjtGeom.mjGEOM_BOX
      np.testing.assert_allclose(geom.pos, pos)
      np.testing.assert_allclose(geom.size, size)


@pytest.mark.parametrize("finger", [1, 2])
def test_finger_sdf_surface_and_gap(finger):
  asset = grasp.SCENE.parent / f"meshes/gripper_link{finger}.stl"
  spec = mujoco.MjSpec.from_string(f'''<mujoco>
    <option sdf_initpoints="40" sdf_iterations="10"/>
    <asset><mesh name="finger" file="{asset}"/></asset>
    <worldbody>
      <geom type="sdf" mesh="finger"/>
      <body pos="1 0 0"><freejoint/><geom type="sphere" size=".0005" mass=".001"/></body>
    </worldbody>
  </mujoco>''')
  spec.meshes[0].octree_maxdepth = 8
  model = spec.compile()
  data = mujoco.MjData(model)
  # The old tip box fills the central recess; the full mesh leaves it open.
  for point, expected in (([0, -0.05, 0.01], False), ([0.018, -0.05, 0.008], True), ([0.04, -0.05, 0.01], False)):
    data.qpos[:3] = point
    mujoco.mj_forward(model, data)
    assert bool(data.ncon) == expected
    if wp.is_cuda_available():
      with wp.ScopedDevice("cuda:0"):
        m = mjw.put_model(model)
        d = mjw.put_data(model, data, nworld=1, nconmax=128, njmax=256)
        mjw.collision(m, d)
        count = int(d.nacon.numpy()[0])
        assert bool(count) == expected
        assert np.isfinite(d.contact.dist.numpy()[:count]).all()


@pytest.mark.parametrize("gpu", [False, True], ids=["cpu", "warp"])
def test_sdf_concavity_and_surface(probe_model, gpu):
  if gpu and not wp.is_cuda_available():
    pytest.skip("CUDA device required for Warp SDF validation")
  model = probe_model
  data = mujoco.MjData(model)
  # The first point is inside the convex hull but in an actual concavity.
  for point, expected in [([0, 0, 0.035], False), ([0, -0.02, 0.01], True), ([0.04, 0, 0.03], False)]:
    data.qpos[:3] = point
    mujoco.mj_forward(model, data)
    assert bool(data.ncon) == expected
    if gpu:
      with wp.ScopedDevice("cuda:0"):
        m = mjw.put_model(model)
        d = mjw.put_data(model, data, nworld=1, nconmax=128, njmax=256)
        mjw.collision(m, d)
        count = int(d.nacon.numpy()[0])
        assert bool(count) == expected
        if expected:
          assert count == data.ncon
          np.testing.assert_allclose(np.median(d.contact.dist.numpy()[:count]), np.median(data.contact.dist), atol=2e-6)
          normals = data.contact.frame.reshape(-1, 3, 3)[:, 0].mean(axis=0)
          np.testing.assert_allclose(d.contact.frame.numpy()[:count, 0].mean(axis=0), normals, atol=2e-4)
  try:
    model.geom_type[0] = mujoco.mjtGeom.mjGEOM_MESH
    data.qpos[:3] = [0, 0, 0.035]
    mujoco.mj_forward(model, data)
    assert data.ncon > 0  # A convex replacement incorrectly fills the gap.
  finally:
    model.geom_type[0] = mujoco.mjtGeom.mjGEOM_SDF


@pytest.mark.parametrize("engine", ["c", "warp"])
def test_complete_transport(model, engine):
  trajectory = grasp.make_trajectory(model)
  if engine == "warp":
    if not wp.is_cuda_available():
      pytest.skip("CUDA device required for Warp grasp validation")
    from grasp_warp import rollout_warp

    trace = rollout_warp(model, trajectory)
  else:
    trace = grasp.rollout_cpu(model, trajectory)
  result = grasp.validate(model, trajectory, trace)
  assert result["passed"], result
  # Validation must reject a failed lift, even with relaxed placement accuracy.
  failed = dict(trace, qpos=trace["qpos"].copy())
  free = model.joint("cat_free").qposadr[0]
  failed["qpos"][:, free + 2] = 0.75
  assert not grasp.validate(model, trajectory, failed)["passed"]


@pytest.mark.parametrize("finger_collision", ["box", "sdf"])
def test_cube_uses_mesh_octree(finger_collision):
  model = grasp.build_model({"finger_collision": finger_collision})
  geom = model.geom("cat_sdf")
  mesh = model.mesh("grasp_cube")
  assert geom.type[0] == mujoco.mjtGeom.mjGEOM_SDF
  assert model.geom_plugin[geom.id] == -1 and model.nplugin == 0
  assert geom.dataid[0] == mesh.id
  assert mesh.vertnum[0] == 8 and mesh.facenum[0] == 12
  start, count = model.mesh_octadr[mesh.id], model.mesh_octnum[mesh.id]
  assert count > 0 and model.oct_depth[start : start + count].max() == 8
  body = model.body("cat_phone_stand")
  assert body.mass[0] == pytest.approx(0.1)
  np.testing.assert_allclose(body.inertia, np.full(3, 0.1 * 0.06**2 / 6), rtol=1e-6)
  assert not np.any(model.geom_type[model.geom_bodyid == body.id] == mujoco.mjtGeom.mjGEOM_BOX)

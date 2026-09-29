"""Exercise STL geometry, uploaded grasp assets, and the authenticated upload API."""

import http.client
import io
import json
import threading
from types import SimpleNamespace

import grasp
import grasp_dashboard
import mujoco
import numpy as np
import pytest
from grasp_config import validate_config
from object_upload import MAX_UPLOAD
from object_upload import ObjectStore
from object_upload import prepare_mesh
from prepare_cat import STL_RECORD


def box_stl(dimensions=(60, 60, 60), ascii=False):
  vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
  vertices = (vertices * dimensions + [123, -45, 6]).astype(np.float32)
  faces = np.array(
    [
      [0, 2, 1],
      [0, 3, 2],
      [4, 5, 6],
      [4, 6, 7],
      [0, 1, 5],
      [0, 5, 4],
      [1, 2, 6],
      [1, 6, 5],
      [2, 3, 7],
      [2, 7, 6],
      [3, 0, 4],
      [3, 4, 7],
    ]
  )
  if ascii:
    facets = [
      "facet normal 0 0 0\nouter loop\n"
      + "\n".join("vertex " + " ".join(map(str, p)) for p in vertices[f])
      + "\nendloop\nendfacet"
      for f in faces
    ]
    return ("solid box\n" + "\n".join(facets) + "\nendsolid box\n").encode()
  records = np.zeros(len(faces), dtype=STL_RECORD)
  records["vertices"] = vertices[faces]
  return b"solid binary box".ljust(80, b" ") + len(faces).to_bytes(4, "little") + records.tobytes()


@pytest.mark.parametrize("ascii", [False, True])
def test_stl_conversion_units_size_and_surface(ascii):
  raw = box_stl((30, 60, 45), ascii=ascii)
  vertices, faces, info = prepare_mesh(raw, "mm", 0)
  np.testing.assert_allclose(info["dimensions_mm"], [30, 60, 45])
  np.testing.assert_allclose(vertices.min(0), [-0.015, -0.03, 0])
  np.testing.assert_allclose(vertices.max(0), [0.015, 0.03, 0.045])
  assert len(faces) == 12 and len(vertices) == 8
  scaled, _, info = prepare_mesh(raw, "mm", 30)
  np.testing.assert_allclose(info["dimensions_mm"], [15, 30, 22.5])
  np.testing.assert_allclose(scaled, vertices / 2)
  # Explicit scaling gives the same simulation geometry regardless of source units.
  metre_vertices, _, _ = prepare_mesh(raw, "m", 60)
  np.testing.assert_allclose(metre_vertices, vertices)


def test_stl_rejects_incomplete_open_and_nonfinite_meshes():
  raw = box_stl()
  with pytest.raises(ValueError):
    prepare_mesh(raw[:-1], "mm", 60)
  with pytest.raises(ValueError, match="封闭"):
    prepare_mesh(raw[:80] + (11).to_bytes(4, "little") + raw[84:-50], "mm", 60)
  invalid = bytearray(raw)
  invalid[96:100] = np.float32(np.nan).tobytes()
  with pytest.raises(ValueError, match="非有限"):
    prepare_mesh(bytes(invalid), "mm", 60)
  with pytest.raises(ValueError, match="格式"):
    prepare_mesh(box_stl(ascii=True).replace(b"endfacet", b"badfacet", 1), "mm", 60)
  with pytest.raises(ValueError, match="X 宽度"):
    prepare_mesh(box_stl((100, 60, 60)), "mm", 0)


def test_uploaded_model_uses_sdf_mesh_and_size_based_waypoints(tmp_path):
  store = ObjectStore(tmp_path)
  asset = store.upload(box_stl((40, 60, 30)), "custom.stl", size_mm=0)
  config = validate_config({"engine": "c", "object_shape": "uploaded", "object_id": asset["id"], "sdf_depth": 5})
  model = grasp.build_model(config, object_path=store.root / asset["id"] / "object.obj")
  assert model.geom("cat_sdf").type[0] == mujoco.mjtGeom.mjGEOM_SDF
  assert model.mesh("uploaded_object").facenum[0] == 12
  np.testing.assert_allclose(grasp.object_bounds(model), [[-0.02, -0.03, 0], [0.02, 0.03, 0.03]], atol=1e-8)
  trajectory = grasp.make_trajectory(model, config)
  assert trajectory["ctrl"][0, model.actuator("gripper_opening").id] == pytest.approx(0.06)
  data = mujoco.MjData(model)
  data.qpos[:] = model.key_qpos[0]
  step = round(3 / model.opt.timestep)  # Original 6 s grasp phase in a 12 s task.
  for joint, actuator in zip(grasp.ARM_JOINTS, grasp.ARM_ACTUATORS, strict=True):
    data.qpos[model.joint(joint).qposadr[0]] = trajectory["ctrl"][step, model.actuator(actuator).id]
  mujoco.mj_kinematics(model, data)
  assert data.site_xpos[model.site("tool_center").id, 2] == pytest.approx(0.766, abs=1e-5)
  with pytest.raises(ValueError, match="请先上传"):
    validate_config({"object_shape": "uploaded"})
  with pytest.raises(ValueError, match="编号"):
    validate_config({"object_shape": "uploaded", "object_id": "../outside"})


def test_upload_api_and_run_snapshot_survive_asset_changes(tmp_path, monkeypatch):
  manager = grasp_dashboard.RunManager(tmp_path)
  monkeypatch.setattr(manager, "_watch", lambda process, run_id: None)
  monkeypatch.setattr(
    grasp_dashboard.multiprocessing,
    "get_context",
    lambda _: SimpleNamespace(Process=lambda **kw: SimpleNamespace(start=lambda: None, is_alive=lambda: False)),
  )
  server = grasp_dashboard.http.server.ThreadingHTTPServer(("127.0.0.1", 0), grasp_dashboard.DashboardHandler)
  server.manager, server.token = manager, "test-token"
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
  auth = {"Cookie": "piper_token=test-token"}

  def request(method, path, body=None, headers=None):
    payload = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None
    connection.request(method, path, body=payload, headers=auth if headers is None else headers)
    response = connection.getresponse()
    return response.status, json.loads(response.read())

  try:
    assert request("POST", "/api/objects?filename=box.stl", box_stl(), headers={})[0] == 401
    assert request("POST", "/api/objects?filename=box.stl", b"bad stl")[0] == 400
    assert request("POST", "/api/objects?filename=box.stl", b"", {**auth, "Content-Length": str(MAX_UPLOAD + 1)})[0] == 400
    status, asset = request("POST", "/api/objects?filename=../../box.stl&unit=mm&size_mm=60", box_stl())
    assert status == 201 and asset["name"] == "box.stl"
    assert request("GET", "/api/objects")[1] == [asset]
    config = {"engine": "c", "object_shape": "uploaded", "object_id": asset["id"]}
    status, created = request("POST", "/api/runs", config)
    assert status == 202
    directory = tmp_path / created["id"]
    saved_mesh = (directory / "object.obj").read_bytes()
    # An existing run carries its own mesh, independent of the upload catalogue.
    (manager.objects.root / asset["id"] / "object.obj").unlink()
    status, run = request("GET", "/api/runs/" + created["id"])
    assert status == 200 and run["object_asset"] == asset
    model = grasp.build_model(run["config"], object_path=directory / "object.obj")
    assert model.mesh("uploaded_object").facenum[0] == 12
    assert (directory / "object.obj").read_bytes() == saved_mesh
    assert request("POST", "/api/runs", config)[0] == 400
  finally:
    connection.close()
    server.shutdown()
    server.server_close()
    thread.join()


def test_uploaded_object_preview_and_run_replay(tmp_path):
  from PIL import Image

  store = ObjectStore(tmp_path)
  asset = store.upload(box_stl(), "preview.stl")
  settings = grasp_dashboard.replay_settings({"width": ["640"]}, duration=0)
  settings.update(x=-0.16, y=-0.075, z=0.78, distance=0.3)
  renderer = grasp_dashboard.ReplayRenderer(tmp_path)
  try:
    frame, _ = renderer.preview(asset["id"], settings)
    assert Image.open(io.BytesIO(frame)).size == (640, 480)
    run_id = "e" * 32
    directory = tmp_path / run_id
    directory.mkdir()
    (directory / "object.obj").write_bytes((store.root / asset["id"] / "object.obj").read_bytes())
    config = validate_config({"object_shape": "uploaded", "object_id": asset["id"], "sdf_depth": 5})
    grasp_dashboard.write_json(directory / "config.json", config)
    model = grasp.build_model(config, object_path=directory / "object.obj")
    np.savez_compressed(directory / "trace.npz", qpos=model.key_qpos[:1], qvel=np.zeros((1, model.nv)))
    replay, _ = renderer.render(run_id, settings)
    assert frame == replay
  finally:
    renderer.close()

"""Focused tests for experiment settings and the local API."""

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
from grasp_config import DEFAULTS
from grasp_config import history_config
from grasp_config import timing_settings
from grasp_config import validate_config


def test_config_validation():
  config = validate_config({"engine": "c", "cat_mass": 0.2})
  assert config["cat_mass"] == 0.2
  assert config["arm_kp"] == DEFAULTS["arm_kp"]
  for invalid in (
    {"unused": 1},
    {"cat_mass": -1},
    {"cat_mass": float("nan")},
    {"arm_kp": [1, 2]},
    {"sdf_depth": 12},
    {"condim": 2},
    {"condim": 5},
    {"cat_torsional_friction": -0.1},
    {"table_rolling_friction": 4},
    {"nworld": 0},
    {"nworld": 32.0},
    {"nworld": True},
    {"nworld": 513},
    {"nworld": 2048},
    {"engine": "c", "nworld": 16},
    {"finger_collision": "mesh"},
    {"finger_collision": 1},
  ):
    with pytest.raises(ValueError):
      validate_config(invalid)
  assert validate_config({})["condim"] == 6
  assert validate_config({})["cat_contact_time"] == 0.004
  assert validate_config({})["table_contact_time"] == 0.004
  assert validate_config({})["cat_torsional_friction"] == 0.005
  assert validate_config({})["table_rolling_friction"] == 0.0001
  assert validate_config({"condim": 3})["condim"] == 3
  assert validate_config({})["nworld"] == 1
  assert validate_config({})["finger_collision"] == "sdf"
  assert validate_config({})["sdf_mode"] == "dense"
  assert validate_config({})["timestep"] == 0.0005
  assert validate_config({})["control_hz"] == 2000
  assert validate_config({})["duration"] == 12
  assert validate_config({"engine": "c"})["sdf_mode"] == "octree"
  with pytest.raises(ValueError, match="稠密 SDF"):
    validate_config({"engine": "c", "sdf_mode": "dense"})
  assert validate_config({"finger_collision": "box"})["finger_collision"] == "box"
  for nworld in (16, 32, 64, 128, 256, 512, 1024):
    assert validate_config({"engine": "warp", "nworld": nworld})["nworld"] == nworld
    with pytest.raises(ValueError, match="批量仿真需要"):
      validate_config({"engine": "c", "nworld": nworld})


def test_configured_timing_holds_targets_and_rescales_validation_phase():
  assert grasp.build_model({"sdf_depth": 5}).opt.timestep == 0.001
  config = validate_config({"engine": "c", "timestep": 0.002, "control_hz": 100, "duration": 12, "sdf_depth": 5})
  model = grasp.build_model(config)
  trajectory = grasp.make_trajectory(model, config)
  assert len(trajectory["ctrl"]) == 6000
  assert trajectory["control_steps"] == 5 and trajectory["effective_control_hz"] == 100
  assert trajectory["duration"] == 12
  np.testing.assert_allclose(trajectory["phase_times"], trajectory["times"] * 2)
  groups = trajectory["ctrl"].reshape(-1, 5, model.nu)
  np.testing.assert_array_equal(groups, np.repeat(groups[:, :1], 5, axis=1))
  assert np.any(np.diff(groups[:, 0], axis=0) != 0)
  native = grasp.make_trajectory(model)
  np.testing.assert_allclose(trajectory["ctrl"][::5], native["ctrl"][::10][:1200], atol=1e-12)
  settings = timing_settings({"timestep": 0.002, "control_hz": 2000, "duration": 6.0012})
  assert settings["effective_control_hz"] == 500
  assert settings["duration"] == pytest.approx(6.002)


def test_history_keeps_native_timing_and_octree_defaults():
  old = history_config({"timestep": 0.002})
  assert old["control_hz"] == 500 and old["duration"] == 24 and old["sdf_mode"] == "octree"
  current = history_config({"duration": 36, "control_hz": 100, "sdf_mode": "dense"})
  assert current["duration"] == 36 and current["control_hz"] == 100 and current["sdf_mode"] == "dense"
  assert grasp_dashboard.replay_settings({"time": ["36"]}, duration=36)["time"] == 36
  with pytest.raises(ValueError):
    grasp_dashboard.replay_settings({"time": ["13"]}, duration=12)


@pytest.mark.parametrize("nworld", [1, 128, 256])
def test_performance_statistics_use_batch_steps_and_physim_environment_clock(nworld):
  config = validate_config({"nworld": nworld, "control_hz": 100})
  metrics = {"simulation_seconds": 3.0, "physics_seconds": 2.0, "physics_steps_per_world": 24000}
  stages = [{"state": "completed", "duration_seconds": seconds} for seconds in (1.0, 3.0, 1.0)]
  stats = grasp_dashboard.performance_statistics(config, metrics, stages)
  assert stats["timing_scope"] == "stepping"
  assert stats["seconds"] == 2 and stats["simulation_seconds"] == 3 and stats["total_seconds"] == 5
  assert stats["physics_steps_per_world"] == 24000 and stats["physics_steps_total"] == nworld * 24000
  assert stats["physics_steps_per_second"] == nworld * 12000
  assert stats["environment_steps_per_world"] == 120 and stats["environment_steps_per_second"] == nworld * 60
  assert stats["control_updates_per_world"] == 1200
  assert stats["effective_control_hz"] == 100
  assert stats["realtime_factor"] == 6
  assert stats["ms_per_batch_physics_step"] == pytest.approx(1 / 12)


def test_legacy_history_statistics_preserve_endpoint_step_and_timing_scope(tmp_path):
  run_id = "f" * 32
  directory = tmp_path / run_id
  directory.mkdir()
  grasp_dashboard.write_json(directory / "config.json", {"nworld": 256, "timestep": 0.001})
  metrics = {"simulation_seconds": 40.0, "world_steps_per_second": 256 * 24001 / 40}
  grasp_dashboard.write_json(directory / "metrics.json", metrics)
  grasp_dashboard.write_json(directory / "status.json", {"id": run_id, "state": "completed", "created_at": "2026-09-29"})
  manager = grasp_dashboard.RunManager(tmp_path)
  history = manager.list_runs()[0]
  stats = history["performance"]
  assert stats["timing_scope"] == "simulation_with_setup" and stats["seconds"] == 40
  assert stats["physics_steps_per_world"] == 24001
  assert stats["duration"] == pytest.approx(24.001)
  assert stats["environment_steps_per_world"] == 240
  assert stats["realtime_factor"] == pytest.approx(24.001 / 40)
  assert manager.get(run_id)["performance"] == stats
  assert "total_seconds" not in stats
  for unavailable in ({}, {"simulation_seconds": 0}, {"simulation_seconds": float("nan")}):
    assert grasp_dashboard.performance_statistics(history_config({}), unavailable) is None


def test_model_overrides():
  config = validate_config(
    {
      "cat_mass": 0.2,
      "cat_friction": 1.1,
      "cat_torsional_friction": 0.012,
      "cat_rolling_friction": 0.0003,
      "table_friction": 0.6,
      "table_torsional_friction": 0.016,
      "table_rolling_friction": 0.0006,
      "cat_contact_time": 0.008,
      "arm_kp": [81, 140, 120, 35, 22, 12],
      "arm_damping": [0.2] * 6,
      "gripper_force_limit": 15,
      "gravcomp": 0.5,
      "sdf_depth": 5,
      "sdf_initpoints": 20,
      "timestep": 0.002,
      "solver_iterations": 60,
      "nconmax": 128,
      "njmax": 512,
    }
  )
  model = grasp.build_model(config)
  assert model.body("cat_phone_stand").mass == pytest.approx(0.2)
  assert model.geom("cat_sdf").friction == pytest.approx([1.1, 0.012, 0.0003])
  assert model.geom("tabletop").friction == pytest.approx([0.6, 0.016, 0.0006])
  assert model.geom("cat_sdf").solref[0] == pytest.approx(0.008)
  assert model.geom("tabletop").solref == pytest.approx([0.004, 1])
  assert model.actuator("position1").gainprm[0] == pytest.approx(81)
  assert model.actuator("gripper_opening").forcerange[1] == pytest.approx(15)
  assert model.dof_damping[model.joint("joint1").dofadr[0]] == pytest.approx(0.2)
  assert model.body_gravcomp[model.body("link1").id] == pytest.approx(0.5)
  assert model.opt.timestep == pytest.approx(0.002)
  assert model.opt.sdf_initpoints == 20 and model.opt.iterations == 60
  assert model.nconmax == 128 and model.njmax == 512
  assert np.all(model.geom_condim == 6)
  data = mujoco.MjData(model)
  data.qpos[:] = model.key_qpos[0]
  data.ctrl[:] = model.key_ctrl[0]
  for _ in range(100):
    mujoco.mj_step(model, data)
    if data.ncon:
      break
  assert data.ncon > 0 and all(contact.dim == 6 for contact in data.contact)
  cat, table = model.geom("cat_sdf").id, model.geom("tabletop").id
  table_contacts = [contact for contact in data.contact if set(contact.geom) == {cat, table}]
  assert table_contacts
  assert table_contacts[0].friction == pytest.approx([0.6, 0.6, 0.016, 0.0006, 0.0006])
  mesh = model.mesh("grasp_cube").id
  start, count = model.mesh_octadr[mesh], model.mesh_octnum[mesh]
  assert model.oct_depth[start : start + count].max() == 5
  assert grasp.make_trajectory(model)["ctrl"].shape[0] == 12001


def test_acceptance_threshold_boundaries():
  model = grasp.build_model({"sdf_depth": 5})
  trajectory = {"times": np.array([10.5, 14, 23, 24])}
  qpos = np.repeat(model.key_qpos[:1], 4, axis=0)
  free = model.joint("cat_free").qposadr[0]
  qpos[:, free : free + 3] = [*grasp.PLACE, 0.9]
  qpos[2:, free + 2] = grasp.TABLE_HEIGHT
  trace = {
    "qpos": qpos,
    "qvel": np.zeros((4, model.nv)),
    "force": np.zeros((4, model.nu)),
    "contacts": np.tile([1, 1, 1, 0], (4, 1)),
    "warnings": np.zeros(mujoco.mjtWarning.mjNWARNING, dtype=int),
  }
  # Isolate threshold comparisons with synthetic states; this is not a rollout.
  assert grasp.validate(model, trajectory, trace)["passed"]
  config = {"minimum_contact_fraction": 1, "maximum_final_drift": 0}
  result = grasp.validate(model, trajectory, trace, config)
  assert result["carried_contact_fraction"] == 1 and result["final_drift_m"] == 0
  assert result["passed"]
  trace["contacts"][0, 0] = 0
  assert not grasp.validate(model, trajectory, trace, config)["passed"]
  trace["contacts"][0, 0] = 1
  trace["qpos"][2, free] += 0.002
  assert not grasp.validate(model, trajectory, trace, config)["passed"]


def test_condim_applies_to_model_and_contacts():
  model = grasp.build_model({"condim": 3, "sdf_depth": 5})
  assert np.all(model.geom_condim == 3)
  data = mujoco.MjData(model)
  data.qpos[:] = model.key_qpos[0]
  data.ctrl[:] = model.key_ctrl[0]
  for _ in range(100):
    mujoco.mj_step(model, data)
    if data.ncon:
      break
  assert data.ncon > 0 and all(contact.dim == 3 for contact in data.contact)


class FakeProcess:
  def __init__(self):
    self.event = threading.Event()
    self.exitcode = None

  def start(self):
    pass

  def is_alive(self):
    return not self.event.is_set()

  def terminate(self):
    self.exitcode = -15
    self.event.set()

  def kill(self):
    self.terminate()

  def join(self, timeout=None):
    self.event.wait(timeout)


def test_authenticated_api_busy_cancel_and_history(tmp_path, monkeypatch):
  monkeypatch.setattr(
    grasp_dashboard.multiprocessing, "get_context", lambda _: SimpleNamespace(Process=lambda **kwargs: FakeProcess())
  )
  manager = grasp_dashboard.RunManager(tmp_path)
  server = grasp_dashboard.http.server.ThreadingHTTPServer(("127.0.0.1", 0), grasp_dashboard.DashboardHandler)
  server.manager, server.token = manager, "test-token"
  server.replay = SimpleNamespace(render=lambda run_id, settings: (b"jpeg-data", {}))
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  connection = http.client.HTTPConnection("127.0.0.1", server.server_port)

  def request(method, path, body=None, headers=None):
    connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=headers or {})
    response = connection.getresponse()
    data = response.read()
    return response.status, json.loads(data), response.getheader("Set-Cookie")

  try:
    status, _, _ = request("GET", "/api/fields")
    assert status == 401
    status, _, cookie = request("POST", "/api/login", {"token": "test-token"})
    assert status == 200
    headers = {"Cookie": cookie.split(";", 1)[0], "Content-Type": "application/json"}
    status, _, _ = request("POST", "/api/runs", {"engine": "c"}, {**headers, "Origin": "http://elsewhere"})
    assert status == 403
    status, _, _ = request("POST", "/api/runs", {"unknown": 1}, headers)
    assert status == 400
    status, run, _ = request("POST", "/api/runs", {"engine": "c"}, headers)
    assert status == 202
    run_id = run["id"]
    status, _, _ = request("POST", "/api/runs", {"engine": "warp"}, headers)
    assert status == 409
    status, _, _ = request("POST", f"/api/runs/{run_id}/cancel", {}, headers)
    assert status == 200
    status, record, _ = request("GET", f"/api/runs/{run_id}", headers=headers)
    assert status == 200 and record["state"] == "canceled"
    assert record["steps"][0]["state"] == "canceled"
    assert record["steps"][0]["duration_seconds"] >= 0
    np.savez_compressed(tmp_path / run_id / "trace.npz", qpos=np.zeros((1, 1)), qvel=np.zeros((1, 1)))
    connection.request("GET", f"/api/runs/{run_id}/frame?width=640&time=1", headers=headers)
    response = connection.getresponse()
    assert response.status == 200 and response.read() == b"jpeg-data"
    status, _, _ = request("GET", f"/api/runs/{run_id}/frame?width=500", headers=headers)
    assert status == 400
    status, _, _ = request("GET", f"/api/runs/{run_id}/frame?world=2", headers=headers)
    assert status == 400
    assert grasp_dashboard.RunManager(tmp_path).list_runs()[0]["id"] == run_id
    status, _, _ = request("POST", "/api/runs", {"engine": "c", "nworld": 16}, headers)
    assert status == 400
    for nworld in (128, 512, 1024):
      status, batch, _ = request("POST", "/api/runs", {"engine": "warp", "nworld": nworld}, headers)
      assert status == 202
      status, batch, _ = request("GET", f"/api/runs/{batch['id']}", headers=headers)
      assert status == 200 and batch["nworld"] == nworld and batch["config"]["nworld"] == nworld
      status, history, _ = request("GET", "/api/runs", headers=headers)
      assert status == 200 and history[0]["nworld"] == nworld
      status, _, _ = request("POST", f"/api/runs/{batch['id']}/cancel", {}, headers)
      assert status == 200
  finally:
    connection.close()
    manager.stop()
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("nworld", [16, 512, 1024])
def test_replay_renders_saved_state(tmp_path, monkeypatch, nworld):
  import mujoco
  from PIL import Image

  model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><body><joint type="slide" axis="1 0 0"/>'
    '<geom type="sphere" size="0.2" rgba="1 0 0 1"/></body></worldbody></mujoco>'
  )
  model_configs = []

  def build_replay_model(config):
    model_configs.append(config)
    return model

  monkeypatch.setattr(grasp, "build_model", build_replay_model)
  run_id = "a" * 32
  directory = tmp_path / run_id
  directory.mkdir()
  grasp_dashboard.write_json(directory / "config.json", {"sdf_depth": 10, "nworld": nworld})
  np.savez_compressed(directory / "trace.npz", qpos=np.array([[0.0], [0.5]]), qvel=np.zeros((2, 1)))
  np.savez_compressed(
    directory / "contact_forces.npz",
    contact_steps=np.array([0]),
    contact_positions=np.array([[[0.0, 0.0, 0.2]]]),
    contact_forces=np.array([[[0.0, 0.0, 1.0]]]),
    contact_groups=np.array([[1]]),
    contact_counts=np.array([1]),
    contact_totals=np.array([1]),
  )
  replay = grasp_dashboard.ReplayRenderer(tmp_path)
  settings = grasp_dashboard.replay_settings({"width": ["640"], "time": ["0"]})
  first, _ = replay.render(run_id, settings)
  settings["contact_forces"] = True
  with_forces, info = replay.render(run_id, settings)
  assert with_forces != first
  assert info["X-Contact-Count"] == "1"
  assert info["X-Contact-Max-N"] == "1.0000"
  settings["contact_forces"] = False
  settings["time"] = model.opt.timestep
  second, _ = replay.render(run_id, settings)
  assert Image.open(io.BytesIO(first)).size == (640, 360)
  assert first != second
  assert model_configs == [{**history_config({"sdf_depth": 10, "nworld": nworld}), "sdf_depth": 5}]
  other_world = grasp_dashboard.world_directory(directory, nworld)
  other_world.mkdir(parents=True)
  np.savez_compressed(other_world / "trace.npz", qpos=np.array([[-0.5], [-1.0]]), qvel=np.zeros((2, 1)))
  with np.load(directory / "contact_forces.npz") as saved:
    forces = {key: saved[key] for key in grasp_dashboard.CONTACT_KEYS}
  forces["contact_forces"] *= 2
  np.savez_compressed(other_world / "contact_forces.npz", **forces)
  settings["world"] = nworld
  settings["contact_forces"] = True
  other_frame, info = replay.render(run_id, settings)
  assert other_frame != second and info["X-Contact-Max-N"] == "2.0000"
  settings["world"] = 2
  with pytest.raises(ValueError, match="没有保存运动轨迹"):
    replay.render(run_id, settings)
  settings["world"] = nworld + 1
  with pytest.raises(ValueError, match="超出本轮"):
    replay.render(run_id, settings)
  settings["world"] = 1
  legacy = tmp_path / ("b" * 32)
  legacy.mkdir()
  grasp_dashboard.write_json(legacy / "config.json", {"sdf_depth": 8})
  np.savez_compressed(legacy / "trace.npz", qpos=np.zeros((1, 1)), qvel=np.zeros((1, 1)))
  settings["contact_forces"] = True
  with pytest.raises(ValueError, match="没有保存接触力"):
    replay.render("b" * 32, settings)
  replay.close()


def test_contact_force_settings_validation():
  assert grasp_dashboard.replay_settings({"contact_forces": ["1"], "force_scale": ["0.3"]})["contact_forces"]
  for query in ({"contact_forces": ["true"]}, {"force_scale": ["0"]}, {"force_scale": ["nan"]}):
    with pytest.raises(ValueError):
      grasp_dashboard.replay_settings(query)
  for world in (256, 512, 1024):
    assert grasp_dashboard.replay_settings({"world": [str(world)]})["world"] == world
  for world in (["0"], ["1025"], ["1.5"], ["1", "2"]):
    with pytest.raises(ValueError):
      grasp_dashboard.replay_settings({"world": world})


@pytest.mark.parametrize("engine", ["c", "warp"])
@pytest.mark.parametrize("timestep", [0.0005, 0.001])
def test_contact_force_trace_from_real_steps(engine, timestep):
  if engine == "warp":
    import warp as wp

    if not wp.is_cuda_available():
      pytest.skip("CUDA unavailable")
  model = grasp.build_model({"sdf_depth": 5, "timestep": timestep})
  trajectory = {
    "qpos": model.key_qpos[:1].copy(),
    "qvel": np.zeros((1, model.nv)),
    "ctrl": np.tile(model.key_ctrl[0], (round(0.03 / timestep), 1)),
  }
  if engine == "c":
    trace = grasp.rollout_cpu(model, trajectory, record_contact_forces=True)
  else:
    from grasp_warp import rollout_warp

    trace = rollout_warp(model, trajectory, record_contact_forces=True)
  np.testing.assert_array_equal(trace["contact_steps"], np.array([0, 0.01 / timestep, 0.02 / timestep], dtype=int))
  assert trace["physics_seconds"] > 0
  assert trace["contact_counts"][-1] > 0
  assert trace["contact_counts"][-1] == trace["contact_totals"][-1]
  assert np.all(trace["contact_groups"][-1, : trace["contact_counts"][-1]] == 1)
  assert np.max(trace["contact_forces"][-1, :, 2]) > 0


@pytest.mark.parametrize("nworld", [16, 32, 128, 256, 512, 1024])
def test_warp_batch_records_independent_worlds(nworld):
  import warp as wp
  from grasp_warp import rollout_warp

  if not wp.is_cuda_available():
    pytest.skip("CUDA unavailable")
  model = grasp.build_model({"sdf_depth": 5})
  controls = np.tile(model.key_ctrl[0], (30, 1))
  controls[:, 0] += np.linspace(0, 0.03, len(controls))
  trajectory = {"qpos": model.key_qpos[:1].copy(), "qvel": np.zeros((1, model.nv)), "ctrl": controls}
  single = rollout_warp(model, trajectory, record_contact_forces=True)
  batch = rollout_warp(model, trajectory, nworld=nworld, record_contact_forces=True)
  assert batch["qpos"].shape == (nworld, 30, model.nq)
  assert batch["qvel"].shape == (nworld, 30, model.nv)
  assert batch["force"].shape == (nworld, 30, model.nu)
  assert batch["contacts"].shape == (nworld, 30, 4)
  assert batch["warnings"].shape == (nworld,) and not np.any(batch["warnings"])
  for world in range(nworld):
    free = model.joint("cat_free").qposadr[0]
    np.testing.assert_allclose(batch["qpos"][world, :, :free], single["qpos"][:, :free], atol=1e-6)
    np.testing.assert_allclose(batch["qpos"][world, :, free : free + 3], single["qpos"][:, free : free + 3], atol=1e-5)
    # Shared contact ordering can slightly change the settled object's rotation.
    np.testing.assert_allclose(batch["qpos"][world, :, free + 3 :], single["qpos"][:, free + 3 :], atol=1e-3)
    assert np.all(batch["contacts"][world, 20:, 2] > 0)
  # Every world's force replay must agree with its own counts in the shared buffer.
  np.testing.assert_array_equal(batch["contact_totals"], batch["contacts"][:, batch["contact_steps"], :3].sum(axis=2))
  np.testing.assert_array_equal(batch["contact_counts"], batch["contact_totals"])
  assert np.all(batch["contact_counts"][:, -1] > 0)
  assert np.all(batch["contact_forces"][:, -1, :, 2].max(axis=1) > 0)


@pytest.mark.parametrize("engine,nworld", [("c", 1), ("warp", 256), ("warp", 512), ("warp", 1024)])
def test_worker_saves_state_without_encoding_video(tmp_path, monkeypatch, engine, nworld):
  run_id = "b" * 32
  directory = tmp_path / run_id
  directory.mkdir()
  grasp_dashboard.write_json(
    directory / "config.json", {"engine": engine, "nworld": nworld, "duration": 36, "control_hz": 100, "sdf_mode": "octree"}
  )
  started_at = grasp_dashboard.utc_now().isoformat()
  grasp_dashboard.write_json(
    directory / "status.json",
    {"id": run_id, "created_at": started_at, "state": "queued", "steps": grasp_dashboard.new_steps(started_at)},
  )
  monkeypatch.setattr(grasp, "build_model", lambda config: object())
  monkeypatch.setattr(grasp, "make_trajectory", lambda model, config: {"ctrl": np.zeros((2, 1))})

  def rollout(model, trajectory, progress, record_contact_forces, **settings):
    assert record_contact_forces
    if engine == "warp":
      assert settings["nworld"] == nworld
    shape = (2, 1) if nworld == 1 else (nworld, 2, 1)
    states = np.ones(shape) if nworld == 1 else np.broadcast_to(np.arange(1, nworld + 1)[:, None, None], shape)
    force_shape = (1, 1, 3) if nworld == 1 else (nworld, 1, 1, 3)
    count_shape = (1,) if nworld == 1 else (nworld, 1)
    return {
      **{key: states for key in grasp_dashboard.WORLD_TRACE_KEYS},
      "warnings": np.zeros(nworld, dtype=int),
      "contact_steps": np.array([0]),
      "contact_positions": np.zeros(force_shape),
      "contact_forces": np.zeros(force_shape),
      "contact_groups": np.zeros(force_shape[:-1]),
      "contact_counts": np.zeros(count_shape),
      "contact_totals": np.zeros(count_shape),
      "physics_seconds": 1e-7,
    }

  if engine == "c":
    monkeypatch.setattr(grasp, "rollout_cpu", rollout)
  else:
    import grasp_warp

    monkeypatch.setattr(grasp_warp, "rollout_warp", rollout)
  monkeypatch.setattr(grasp, "validate", lambda model, trajectory, trace, config: {"passed": True})
  grasp_dashboard.run_worker(tmp_path, run_id)
  status = grasp_dashboard.read_json(directory / "status.json")
  assert status["state"] == "completed"
  assert status["sim_time"] == 36
  assert status["timing"]["effective_control_hz"] == 100
  assert status["passed_count"] == nworld
  assert status["replay_worlds"] == nworld
  metrics = grasp_dashboard.read_json(directory / "metrics.json")
  assert metrics["nworld"] == nworld and len(metrics["worlds"]) == nworld
  assert metrics["physics_seconds"] == 1e-7 and metrics["physics_steps_per_world"] == 2
  assert metrics["validation_seconds"] >= 0 and metrics["save_seconds"] >= 0
  manager = grasp_dashboard.RunManager(tmp_path)
  assert manager.list_runs()[0]["performance"]["timing_scope"] == "stepping"
  assert manager.list_runs()[0]["performance"]["physics_steps_per_world"] == 2
  assert len(status["steps"]) == 5
  assert all(step["state"] == "completed" and step["duration_seconds"] >= 0 for step in status["steps"])
  with np.load(directory / "trace.npz") as saved:
    np.testing.assert_array_equal(saved["qpos"], np.ones((2, 1)))
  with np.load(directory / "contact_forces.npz") as saved:
    assert set(saved.files) == set(grasp_dashboard.CONTACT_KEYS)
  for world in range(1, nworld + 1):
    destination = grasp_dashboard.world_directory(directory, world)
    with np.load(destination / "trace.npz") as saved:
      np.testing.assert_array_equal(saved["qpos"], np.full((2, 1), world))
    with np.load(destination / "contact_forces.npz") as saved:
      assert saved["contact_positions"].shape == (1, 1, 3)
  assert not (directory / "video.mp4").exists()


def test_parallel_save_preserves_all_arrays_and_waits_for_completion(tmp_path, monkeypatch):
  nworld = 16
  trace = {key: np.arange(nworld * 6, dtype=np.float32).reshape(nworld, 2, 3) for key in grasp_dashboard.WORLD_TRACE_KEYS}
  trace["warnings"] = np.arange(nworld, dtype=np.int32)
  trace.update({key: np.arange(nworld, dtype=np.int32).reshape(nworld, 1) for key in grasp_dashboard.CONTACT_KEYS})
  trace["contact_steps"] = np.array([0], dtype=np.int32)
  original = {key: value.copy() for key, value in trace.items()}
  monkeypatch.setattr(grasp_dashboard.os, "cpu_count", lambda: 8)
  barrier = threading.Barrier(8)
  compress = np.savez_compressed

  def concurrent_compress(path, **values):
    if "contact_positions" in values and values["contact_positions"][0] < 8:
      # All eight jobs must reach compression together, without timing-based speed assertions.
      barrier.wait(timeout=10)
    compress(path, **values)

  monkeypatch.setattr(np, "savez_compressed", concurrent_compress)
  grasp_dashboard.save_world_traces(tmp_path, trace, nworld)
  for world in range(nworld):
    destination = grasp_dashboard.world_directory(tmp_path, world + 1)
    expected = grasp_dashboard.world_trace(trace, nworld, world)
    for filename, keys in (("trace", grasp_dashboard.WORLD_TRACE_KEYS), ("contact_forces", grasp_dashboard.CONTACT_KEYS)):
      with np.load(destination / f"{filename}.npz") as saved:
        assert set(saved.files) == set(keys)
        for key in keys:
          np.testing.assert_array_equal(saved[key], expected[key])
          assert saved[key].dtype == expected[key].dtype
  assert not list(tmp_path.rglob("*.part.npz"))
  for key in trace:
    np.testing.assert_array_equal(trace[key], original[key])


@pytest.mark.parametrize("nworld", [1, 16])
def test_save_failure_propagates_and_preserves_existing_archive(tmp_path, monkeypatch, nworld):
  trace = {key: np.zeros((nworld, 1, 1)) for key in grasp_dashboard.WORLD_TRACE_KEYS + grasp_dashboard.CONTACT_KEYS}
  trace["warnings"] = np.zeros(nworld, dtype=int)
  trace["contact_steps"] = np.zeros(1, dtype=int)
  if nworld == 1:
    trace = {key: value if key in ("warnings", "contact_steps") else value[0] for key, value in trace.items()}
  existing = tmp_path / "contact_forces.npz"
  existing.write_bytes(b"previous archive")
  compress = np.savez_compressed

  def fail_compress(path, **values):
    if path.parent == tmp_path and path.name == "contact_forces.part.npz":
      path.write_bytes(b"incomplete archive")
      raise OSError("disk full")
    compress(path, **values)

  monkeypatch.setattr(np, "savez_compressed", fail_compress)
  with pytest.raises(OSError, match="disk full"):
    grasp_dashboard.save_world_traces(tmp_path, trace, nworld)
  assert existing.read_bytes() == b"previous archive"
  assert not list(tmp_path.rglob("*.part.npz"))


def test_worker_marks_save_failure_without_completing_run(tmp_path, monkeypatch):
  run_id = "a" * 32
  directory = tmp_path / run_id
  directory.mkdir()
  started_at = grasp_dashboard.utc_now().isoformat()
  grasp_dashboard.write_json(directory / "config.json", {"engine": "c"})
  grasp_dashboard.write_json(
    directory / "status.json", {"id": run_id, "state": "queued", "steps": grasp_dashboard.new_steps(started_at)}
  )
  monkeypatch.setattr(grasp, "build_model", lambda config: object())
  monkeypatch.setattr(grasp, "make_trajectory", lambda model, config: {"ctrl": np.zeros((2, 1))})
  monkeypatch.setattr(grasp, "rollout_cpu", lambda *args, **kwargs: {})
  monkeypatch.setattr(grasp_dashboard, "validate_run", lambda *args: {"passed": True, "passed_count": 1})

  def fail_save(*args):
    raise OSError("disk full")

  monkeypatch.setattr(grasp_dashboard, "save_world_traces", fail_save)
  grasp_dashboard.run_worker(tmp_path, run_id)
  status = grasp_dashboard.read_json(directory / "status.json")
  assert status["state"] == "failed" and status["error"] == "disk full"
  assert [step["state"] for step in status["steps"]] == ["completed"] * 4 + ["failed"]
  assert "replay_worlds" not in status


def test_batch_validation_checks_every_world(monkeypatch):
  trace = {key: np.arange(16).reshape(16, 1, 1) for key in grasp_dashboard.WORLD_TRACE_KEYS}
  trace["warnings"] = np.zeros(16, dtype=int)
  trace["warnings"][-1] = 1
  trace.update({key: np.zeros((16, 1)) for key in grasp_dashboard.CONTACT_KEYS})
  trace["contact_steps"] = np.zeros(1)
  validated = []

  def validate(model, trajectory, world_trace, config):
    validated.append(int(world_trace["qpos"][0, 0]))
    return {"passed": not bool(np.any(world_trace["warnings"]))}

  monkeypatch.setattr(grasp, "validate", validate)
  metrics = grasp_dashboard.validate_run(None, None, trace, validate_config({"nworld": 16}))
  assert validated == list(range(16))
  assert not metrics["passed"] and metrics["passed_count"] == 15
  assert metrics["pass_fraction"] == 15 / 16
  assert metrics["worlds"][0]["passed"] and not metrics["worlds"][-1]["passed"]
  replay = grasp_dashboard.world_trace(trace, 16, 15)
  np.testing.assert_array_equal(replay["qpos"], trace["qpos"][15])
  assert replay["warnings"][0] == 1


def test_worker_marks_failed_stage(tmp_path, monkeypatch):
  run_id = "c" * 32
  directory = tmp_path / run_id
  directory.mkdir()
  started_at = grasp_dashboard.utc_now().isoformat()
  grasp_dashboard.write_json(directory / "config.json", {"engine": "c"})
  grasp_dashboard.write_json(
    directory / "status.json", {"id": run_id, "state": "queued", "steps": grasp_dashboard.new_steps(started_at)}
  )

  def fail_compile(config):
    raise RuntimeError("compile failed")

  monkeypatch.setattr(grasp, "build_model", fail_compile)
  grasp_dashboard.run_worker(tmp_path, run_id)
  status = grasp_dashboard.read_json(directory / "status.json")
  assert status["state"] == "failed"
  assert [step["state"] for step in status["steps"]] == ["completed", "failed", "pending", "pending", "pending"]
  assert status["steps"][1]["duration_seconds"] >= 0


def test_oom_worker_failure_is_recorded(tmp_path):
  manager = grasp_dashboard.RunManager(tmp_path)
  run_id = "d" * 32
  directory = tmp_path / run_id
  directory.mkdir()
  started_at = grasp_dashboard.utc_now().isoformat()
  grasp_dashboard.write_json(
    directory / "status.json",
    {"id": run_id, "state": "running", "steps": grasp_dashboard.new_steps(started_at)},
  )
  process = FakeProcess()
  process.exitcode = -9
  process.event.set()
  manager.active, manager.active_id = process, run_id
  manager._watch(process, run_id)
  status = grasp_dashboard.read_json(directory / "status.json")
  assert status["state"] == "failed"
  assert "内存" in status["error"]
  assert status["steps"][0]["state"] == "failed"


def test_gripper_replay_restores_prescribed_root(tmp_path, monkeypatch):
  import mujoco

  model = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><body mocap="true"><geom type="sphere" size="0.1"/></body>'
    '<body><joint type="slide"/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
  )
  monkeypatch.setattr(grasp, "build_model", lambda config: model)
  run_id = "e" * 32
  directory = tmp_path / run_id
  directory.mkdir()
  grasp_dashboard.write_json(directory / "config.json", {"robot_mode": "gripper_only", "reference_run_id": "a" * 32})
  poses = np.array([[0, 0, 0.8, 1, 0, 0, 0], [0.3, 0.1, 0.9, 1, 0, 0, 0]])
  np.savez(directory / "trace.npz", qpos=np.zeros((2, 1)), qvel=np.zeros((2, 1)), root_pose=poses)
  observed = []
  original = mujoco.mj_forward

  def inspect_pose(model, data):
    observed.append(data.mocap_pos.copy())
    original(model, data)

  monkeypatch.setattr(mujoco, "mj_forward", inspect_pose)
  replay = grasp_dashboard.ReplayRenderer(tmp_path)
  try:
    frame, _ = replay.render(run_id, grasp_dashboard.replay_settings({"time": ["1"], "width": ["640"]}))
    assert frame.startswith(b"\xff\xd8")
    np.testing.assert_array_equal(observed[-1][0], poses[1, :3])
  finally:
    replay.close()

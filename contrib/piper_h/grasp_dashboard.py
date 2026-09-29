"""Local web dashboard for repeatable PiPER H grasp experiments."""

import argparse
import concurrent.futures
import contextlib
import datetime
import hmac
import http.cookies
import http.server
import io
import json
import math
import multiprocessing
import os
import queue
import re
import secrets
import signal
import threading
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import parse_qs
from urllib.parse import urlparse

from grasp_config import DEFAULTS
from grasp_config import FIELDS
from grasp_config import history_config
from grasp_config import timing_settings
from grasp_config import validate_config

HERE = Path(__file__).resolve().parent
RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
MAX_BODY = 65536
STAGE_NAMES = ("提交配置", "编译模型", "生成动作轨迹", "运行仿真", "验收并保存")
CONTACT_KEYS = (
  "contact_steps",
  "contact_positions",
  "contact_forces",
  "contact_groups",
  "contact_counts",
  "contact_totals",
)
WORLD_TRACE_KEYS = ("qpos", "qvel", "force", "contacts", "warnings")


def utc_now():
  return datetime.datetime.now(datetime.timezone.utc)


def new_steps(started_at):
  return [
    {"name": name, "state": "running" if index == 0 else "pending", "started_at": started_at if index == 0 else None}
    for index, name in enumerate(STAGE_NAMES)
  ]


def finish_active_step(status, state="completed", now=None):
  """Record elapsed wall time for the currently running stage."""
  now = now or utc_now()
  for step in status.get("steps", []):
    if step["state"] == "running":
      started = datetime.datetime.fromisoformat(step["started_at"])
      step.update(state=state, duration_seconds=round(max(0, (now - started).total_seconds()), 2))
      return


def begin_step(status, index, now=None):
  now = now or utc_now()
  finish_active_step(status, now=now)
  step = status["steps"][index]
  step.update(state="running", started_at=now.isoformat())
  status["phase"] = f"正在{step['name']}"


def data_directory():
  """Return the user-owned history directory."""
  base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
  return base / "mujoco_warp" / "piper_h_runs"


def read_json(path):
  return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
  os.replace(temporary, path)


def world_directory(directory, world):
  """Keep world 1 at the legacy path and store other worlds separately."""
  return directory if world == 1 else directory / "worlds" / f"{world:04d}"


def world_trace(trace, nworld, world):
  """Return one world's states and sampled contact forces without copying the batch."""
  if nworld == 1:
    return dict(trace)
  result = {key: trace[key][world] for key in WORLD_TRACE_KEYS}
  result["warnings"] = trace["warnings"][world : world + 1]
  result["contact_steps"] = trace["contact_steps"]
  result.update({key: trace[key][world] for key in CONTACT_KEYS if key != "contact_steps"})
  return result


def validate_run(model, trajectory, trace, config):
  """Validate every world, returning the aggregate and individual metrics."""
  import grasp

  nworld = config["nworld"]
  worlds = []
  for world in range(nworld):
    worlds.append({"world": world + 1, **grasp.validate(model, trajectory, world_trace(trace, nworld, world), config)})
  passed_count = sum(world["passed"] for world in worlds)
  metrics = {
    **worlds[0],
    "nworld": nworld,
    "passed_count": passed_count,
    "pass_fraction": passed_count / nworld,
    "passed": passed_count == nworld,
    "worlds": worlds,
  }
  return metrics


def save_world_traces(directory, trace, nworld):
  """Save each world's complete state and force replay in its own compressed archives."""
  import numpy as np

  for world in range(nworld):
    destination = world_directory(directory, world + 1)
    destination.mkdir(parents=True, exist_ok=True)
    replay = world_trace(trace, nworld, world)
    contact_trace = {key: replay.pop(key) for key in CONTACT_KEYS}
    for filename, values in (("contact_forces", contact_trace), ("trace", replay)):
      temporary = destination / f"{filename}.part.npz"
      np.savez_compressed(temporary, **values)
      os.replace(temporary, destination / f"{filename}.npz")


def run_worker(root, run_id):
  """Simulate one experiment and save its physical states in a separate process."""
  run_dir = Path(root) / run_id
  status_path = run_dir / "status.json"
  status = read_json(status_path)

  def update(**values):
    status.update(values)
    write_json(status_path, status)

  def stage(index, **values):
    begin_step(status, index)
    update(**values)

  with (run_dir / "log.txt").open("w", encoding="utf-8", buffering=1) as stream:
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
      try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import grasp

        saved_config = read_json(run_dir / "config.json")
        config = validate_config(history_config(saved_config))
        stage(1, state="running")
        model = grasp.build_model(config)
        stage(2)
        trajectory = grasp.make_trajectory(model, config)
        timing = timing_settings(config)
        duration = timing["duration"]
        update(timing=timing)

        def progress(sim_time):
          update(sim_time=round(sim_time, 2))

        stage(3, sim_time=0)
        simulation_started = time.perf_counter()
        if config["engine"] == "warp":
          from grasp_warp import rollout_warp

          trace = rollout_warp(
            model,
            trajectory,
            nworld=config["nworld"],
            nconmax=config["nconmax"],
            njmax=config["njmax"],
            progress=progress,
            record_contact_forces=True,
            sdf_mode=config["sdf_mode"],
          )
        else:
          trace = grasp.rollout_cpu(model, trajectory, progress=progress, record_contact_forces=True)
        simulation_seconds = time.perf_counter() - simulation_started
        stage(4, sim_time=duration)
        metrics = validate_run(model, trajectory, trace, config)
        metrics["simulation_seconds"] = simulation_seconds
        metrics["timing"] = timing
        metrics["sdf_mode"] = config["sdf_mode"]
        metrics["world_steps_per_second"] = config["nworld"] * len(trajectory["ctrl"]) / simulation_seconds
        write_json(run_dir / "metrics.json", metrics)
        save_world_traces(run_dir, trace, config["nworld"])
        finish_active_step(status)
        update(
          state="completed",
          phase="完成",
          sim_time=duration,
          passed=metrics["passed"],
          passed_count=metrics["passed_count"],
          replay_worlds=config["nworld"],
        )
      except Exception as exc:
        traceback.print_exc()
        finish_active_step(status, "failed")
        update(state="failed", phase="失败", error=str(exc))


def replay_settings(query, duration=24):
  """Read only bounded camera and image settings from a frame request."""
  ranges = {
    "time": (0, duration),
    "azimuth": (-360, 360),
    "elevation": (-89, 89),
    "distance": (0.25, 10),
    "x": (-3, 3),
    "y": (-3, 3),
    "z": (-1, 4),
    "force_scale": (0.005, 2),
  }
  defaults = {"time": 0, "azimuth": 140, "elevation": -25, "distance": 3.3, "x": 0, "y": 0, "z": 0.75, "force_scale": 0.3}
  if set(query) - (set(ranges) | {"width", "contact_forces", "world"}):
    raise ValueError("未知的回放参数")
  settings = {}
  for key, (low, high) in ranges.items():
    value = float(query.get(key, [defaults[key]])[0])
    if not math.isfinite(value) or not low <= value <= high or len(query.get(key, [])) > 1:
      raise ValueError(f"{key} 超出有效范围")
    settings[key] = value
  width = query.get("width", [960])
  if len(width) != 1 or width[0] not in ("640", "960", "1280", "1920", 640, 960, 1280, 1920):
    raise ValueError("不支持的渲染分辨率")
  settings["width"] = int(width[0])
  show_forces = query.get("contact_forces", ["0"])
  if len(show_forces) != 1 or show_forces[0] not in ("0", "1"):
    raise ValueError("接触力显示参数无效")
  settings["contact_forces"] = show_forces[0] == "1"
  world = query.get("world", ["1"])
  if len(world) != 1 or not str(world[0]).isdigit() or not 1 <= int(world[0]) <= 256:
    raise ValueError("回放场景无效")
  settings["world"] = int(world[0])
  return settings


class ReplayRenderer:
  """Keep MuJoCo's EGL contexts in one thread and reuse compiled history models."""

  def __init__(self, root):
    self.root = Path(root)
    self.jobs = queue.Queue(maxsize=8)
    self.thread = threading.Thread(target=self._loop, daemon=True)
    self.thread.start()

  def render(self, run_id, settings):
    future = concurrent.futures.Future()
    try:
      self.jobs.put_nowait((run_id, settings, future))
    except queue.Full as exc:
      raise RuntimeError("回放渲染繁忙，请稍后重试") from exc
    return future.result(timeout=90)

  def close(self):
    self.jobs.put(None)
    self.thread.join()

  def _loop(self):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import grasp
    import mujoco
    import numpy as np
    from PIL import Image

    cached = {}
    while True:
      job = self.jobs.get()
      if job is None:
        for item in cached.values():
          if item["renderer"] is not None:
            item["renderer"].close()
        return
      run_id, settings, future = job
      try:
        if run_id not in cached:
          directory = self.root / run_id
          config = history_config(read_json(directory / "config.json"))
          # Replay uses recorded qpos/qvel, so contact resolution cannot change motion.
          # Keep the same visible meshes while avoiding a slow deep SDF rebuild.
          model = grasp.build_model({**config, "sdf_depth": 5})
          cached[run_id] = {
            "model": model,
            "data": mujoco.MjData(model),
            "worlds": {},
            "nworld": config.get("nworld", 1),
            "renderer": None,
            "width": None,
          }
          if len(cached) > 2:
            oldest = next(iter(cached))
            if cached[oldest]["renderer"] is not None:
              cached[oldest]["renderer"].close()
            del cached[oldest]
        item = cached[run_id]
        world = settings["world"]
        if not 1 <= world <= item["nworld"]:
          raise ValueError("回放场景超出本轮场景数")
        if world not in item["worlds"]:
          directory = world_directory(self.root / run_id, world)
          if not (directory / "trace.npz").exists():
            raise ValueError("这个场景没有保存运动轨迹，请重新运行仿真")
          with np.load(directory / "trace.npz") as saved:
            states = {key: saved[key] for key in ("qpos", "qvel")}
          contact_path = directory / "contact_forces.npz"
          states["contact_trace"] = None
          if contact_path.exists():
            with np.load(contact_path) as saved:
              states["contact_trace"] = {key: saved[key] for key in CONTACT_KEYS}
          item["worlds"][world] = states
          if len(item["worlds"]) > 2:
            del item["worlds"][next(iter(item["worlds"]))]
        states = item["worlds"][world]
        model, data = item["model"], item["data"]
        width = settings["width"]
        if item["width"] != width:
          if item["renderer"] is not None:
            item["renderer"].close()
          model.vis.global_.offwidth = width
          model.vis.global_.offheight = width * 3 // 4
          item["renderer"] = mujoco.Renderer(model, width=width, height=width * 3 // 4)
          item["width"] = width
        step = min(round(settings["time"] / model.opt.timestep), len(states["qpos"]) - 1)
        data.qpos[:] = states["qpos"][step]
        data.qvel[:] = states["qvel"][step]
        data.time = step * model.opt.timestep
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = settings["azimuth"]
        camera.elevation = settings["elevation"]
        camera.distance = settings["distance"]
        camera.lookat[:] = [settings[key] for key in ("x", "y", "z")]
        item["renderer"].update_scene(data, camera=camera)
        info = {}
        if settings["contact_forces"]:
          force_trace = states["contact_trace"]
          if force_trace is None:
            raise ValueError("这轮历史运行没有保存接触力，请重新运行仿真")
          sample = int(np.argmin(np.abs(force_trace["contact_steps"] - step)))
          count = int(force_trace["contact_counts"][sample])
          total = int(force_trace["contact_totals"][sample])
          vectors = force_trace["contact_forces"][sample, :count]
          magnitudes = np.linalg.norm(vectors, axis=1)
          colors = {1: (0.2, 0.75, 1, 1), 2: (1, 0.6, 0.15, 1), 3: (1, 0.6, 0.15, 1), 4: (1, 0.3, 0.45, 1)}
          scene = item["renderer"].scene
          for contact, magnitude in enumerate(magnitudes):
            if magnitude < 1e-5 or scene.ngeom >= scene.maxgeom:
              continue
            point = force_trace["contact_positions"][sample, contact].astype(float)
            length = min(float(magnitude) * settings["force_scale"], 0.2)
            tip = point + vectors[contact] * (length / magnitude)
            geom = scene.geoms[scene.ngeom]
            color = np.array(colors[int(force_trace["contact_groups"][sample, contact])], dtype=np.float32)
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), point, np.eye(3).ravel(), color)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.0025, point, tip)
            scene.ngeom += 1
          info = {
            "X-Contact-Count": str(total),
            "X-Contact-Max-N": f"{max(magnitudes, default=0):.4f}",
            "X-Contact-Sum-N": f"{sum(magnitudes):.4f}",
            "X-Contact-Omitted": str(total - count),
            "X-Contact-Sample-Time": f"{(int(force_trace['contact_steps'][sample]) + 1) * model.opt.timestep:.3f}",
          }
        output = io.BytesIO()
        Image.fromarray(item["renderer"].render()).save(output, format="JPEG", quality=90)
        future.set_result((output.getvalue(), info))
      except Exception as exc:
        future.set_exception(exc)


class RunManager:
  """Own one active worker and durable run history."""

  def __init__(self, root):
    self.root = Path(root)
    self.root.mkdir(parents=True, exist_ok=True)
    self.lock = threading.Lock()
    self.active = None
    self.active_id = None
    for directory in self.root.iterdir():
      path = directory / "status.json"
      if directory.is_dir() and path.exists():
        status = read_json(path)
        if status.get("state") in ("queued", "running"):
          finish_active_step(status, "failed")
          status.update(state="failed", phase="中断", error="服务上次退出时仿真尚未完成")
          write_json(path, status)

  def list_runs(self):
    runs = []
    for directory in self.root.iterdir():
      if directory.is_dir() and RUN_ID.fullmatch(directory.name) and (directory / "status.json").exists():
        runs.append(read_json(directory / "status.json"))
    return sorted(runs, key=lambda run: run["created_at"], reverse=True)

  def get(self, run_id):
    if not RUN_ID.fullmatch(run_id):
      return None
    directory = self.root / run_id
    if not (directory / "status.json").exists():
      return None
    run = read_json(directory / "status.json")
    run["config"] = history_config(read_json(directory / "config.json"))
    run["timing"] = timing_settings(run["config"])
    if (directory / "metrics.json").exists():
      run["metrics"] = read_json(directory / "metrics.json")
    run["replay_ready"] = (directory / "trace.npz").exists()
    run["replay_worlds"] = run.get("replay_worlds", 1 if run["replay_ready"] else 0)
    run["contact_forces_ready"] = (directory / "contact_forces.npz").exists()
    run["video_ready"] = (directory / "video.mp4").exists()
    return run

  def start(self, config):
    with self.lock:
      if self.active is not None and self.active.is_alive():
        raise RuntimeError("已有仿真正在运行，请等待或取消")
      run_id = uuid.uuid4().hex
      directory = self.root / run_id
      directory.mkdir()
      write_json(directory / "config.json", config)
      created_at = utc_now().isoformat()
      write_json(
        directory / "status.json",
        {
          "id": run_id,
          "created_at": created_at,
          "state": "queued",
          "phase": "正在提交配置",
          "engine": config["engine"],
          "nworld": config["nworld"],
          "sim_time": 0,
          "steps": new_steps(created_at),
        },
      )
      process = multiprocessing.get_context("spawn").Process(target=run_worker, args=(str(self.root), run_id))
      process.start()
      self.active, self.active_id = process, run_id
      threading.Thread(target=self._watch, args=(process, run_id), daemon=True).start()
      return run_id

  def _watch(self, process, run_id):
    process.join()
    with self.lock:
      path = self.root / run_id / "status.json"
      status = read_json(path)
      if status["state"] in ("queued", "running"):
        finish_active_step(status, "failed")
        error = f"仿真进程异常退出（代码 {process.exitcode}）"
        if process.exitcode == -signal.SIGKILL:
          error = "仿真进程被系统终止，可能是内存不足。请降低并行场景数或 SDF 深度后重试。"
        status.update(state="failed", phase="失败", error=error)
        write_json(path, status)
      if self.active is process:
        self.active, self.active_id = None, None

  def cancel(self, run_id):
    with self.lock:
      if self.active_id != run_id or self.active is None or not self.active.is_alive():
        return False
      process = self.active
      process.terminate()
      process.join(timeout=5)
      if process.is_alive():
        process.kill()
        process.join()
      path = self.root / run_id / "status.json"
      status = read_json(path)
      finish_active_step(status, "canceled")
      status.update(state="canceled", phase="已取消")
      write_json(path, status)
      if self.active is process:
        self.active, self.active_id = None, None
    return True

  def stop(self):
    if self.active_id is not None:
      self.cancel(self.active_id)


class DashboardHandler(http.server.BaseHTTPRequestHandler):
  """Serve the page and a small authenticated same-origin API."""

  def _send(self, code, payload, content_type="application/json; charset=utf-8", headers=None):
    data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
    self.send_response(code)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(data)))
    self.send_header("Cache-Control", "no-store")
    self.send_header("X-Content-Type-Options", "nosniff")
    for key, value in (headers or {}).items():
      self.send_header(key, value)
    self.end_headers()
    self.wfile.write(data)

  def _authorized(self):
    cookie = http.cookies.SimpleCookie()
    try:
      cookie.load(self.headers.get("Cookie", ""))
    except http.cookies.CookieError:
      return False
    value = cookie.get("piper_token")
    return value is not None and hmac.compare_digest(value.value, self.server.token)

  def _origin_valid(self):
    origin = self.headers.get("Origin")
    return origin is None or origin == f"http://{self.headers.get('Host')}"

  def _body(self):
    length = int(self.headers.get("Content-Length", "0"))
    if length < 1 or length > MAX_BODY:
      raise ValueError("请求体大小无效")
    return json.loads(self.rfile.read(length))

  def do_GET(self):
    path = urlparse(self.path).path
    if path == "/":
      if self._authorized():
        self._send(200, (HERE / "grasp_dashboard.html").read_bytes(), "text/html; charset=utf-8")
      else:
        self._send(200, LOGIN_HTML.encode(), "text/html; charset=utf-8")
      return
    if not self._authorized():
      self._send(401, {"error": "请先输入访问令牌"})
      return
    if path == "/api/fields":
      self._send(200, {"fields": FIELDS, "defaults": DEFAULTS, "engines": ["c", "warp"]})
    elif path == "/api/runs":
      self._send(200, self.server.manager.list_runs())
    elif path.startswith("/api/runs/"):
      parts = path.split("/")
      run_id = parts[3] if len(parts) >= 4 else ""
      run = self.server.manager.get(run_id)
      if run is None:
        self._send(404, {"error": "运行记录不存在"})
      elif len(parts) == 4:
        self._send(200, run)
      elif len(parts) == 5 and parts[4] == "frame":
        if not run["replay_ready"]:
          self._send(404, {"error": "仿真状态尚未保存"})
          return
        try:
          settings = replay_settings(parse_qs(urlparse(self.path).query), run["timing"]["duration"])
          if settings["world"] > run["replay_worlds"]:
            raise ValueError("这个场景没有保存运动轨迹，请重新运行仿真")
          frame, info = self.server.replay.render(run_id, settings)
          self._send(200, frame, "image/jpeg", headers=info)
        except ValueError as exc:
          self._send(400, {"error": str(exc)})
        except (RuntimeError, TimeoutError) as exc:
          self._send(503, {"error": str(exc)})
      elif len(parts) == 5 and parts[4] == "video":
        self._send_video(self.server.manager.root / run_id / "video.mp4")
      elif len(parts) == 5 and parts[4] == "log":
        log = self.server.manager.root / run_id / "log.txt"
        self._send(200, log.read_bytes() if log.exists() else b"", "text/plain; charset=utf-8")
      else:
        self._send(404, {"error": "路径不存在"})
    else:
      self._send(404, {"error": "路径不存在"})

  def _send_video(self, path):
    if not path.exists():
      self._send(404, {"error": "视频尚未生成"})
      return
    size = path.stat().st_size
    start, end = 0, size - 1
    request_range = self.headers.get("Range")
    if request_range:
      match = re.fullmatch(r"bytes=(\d+)-(\d*)", request_range)
      if not match:
        self._send(416, {"error": "不支持的 Range"})
        return
      start = int(match.group(1))
      end = int(match.group(2)) if match.group(2) else end
      if start >= size or end < start:
        self._send(416, {"error": "Range 超出视频长度"})
        return
      end = min(end, size - 1)
    self.send_response(206 if request_range else 200)
    self.send_header("Content-Type", "video/mp4")
    self.send_header("Content-Length", str(end - start + 1))
    self.send_header("Accept-Ranges", "bytes")
    self.send_header("X-Content-Type-Options", "nosniff")
    if request_range:
      self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    self.end_headers()
    with path.open("rb") as stream:
      stream.seek(start)
      remaining = end - start + 1
      while remaining:
        chunk = stream.read(min(65536, remaining))
        if not chunk:
          break
        try:
          self.wfile.write(chunk)
        except BrokenPipeError:
          break
        remaining -= len(chunk)

  def do_POST(self):
    path = urlparse(self.path).path
    if not self._origin_valid():
      self._send(403, {"error": "请求来源不匹配"})
      return
    try:
      body = self._body()
      if path == "/api/login":
        token = body.get("token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not hmac.compare_digest(token, self.server.token):
          self._send(401, {"error": "访问令牌不正确"})
          return
        self._send(200, {"ok": True}, headers={"Set-Cookie": f"piper_token={token}; HttpOnly; SameSite=Strict; Path=/"})
        return
      if not self._authorized():
        self._send(401, {"error": "请先输入访问令牌"})
        return
      if path == "/api/runs":
        config = validate_config(body)
        run_id = self.server.manager.start(config)
        self._send(202, {"id": run_id})
      elif len(path.split("/")) == 5 and path.startswith("/api/runs/") and path.endswith("/cancel"):
        run_id = path.split("/")[3]
        if not RUN_ID.fullmatch(run_id):
          self._send(404, {"error": "运行记录不存在"})
        elif self.server.manager.cancel(run_id):
          self._send(200, {"canceled": True})
        else:
          self._send(409, {"error": "该运行已结束或不是当前运行"})
      else:
        self._send(404, {"error": "路径不存在"})
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
      self._send(400, {"error": str(exc)})
    except RuntimeError as exc:
      self._send(409, {"error": str(exc)})


LOGIN_HTML = """<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>PiPER H 登录</title><style>body{font:16px system-ui;background:#111827;color:#f9fafb;display:grid;place-items:center;
min-height:100vh;margin:0}main{background:#1f2937;padding:2rem;border-radius:14px;width:min(90vw,380px)}input,button{
box-sizing:border-box;width:100%;padding:.75rem;margin:.5rem 0;border-radius:8px;font:inherit}button{background:#38bdf8;border:0;
cursor:pointer}input{background:#111827;color:white;border:1px solid #6b7280}</style><main><h1>PiPER H 仿真实验</h1>
<p>请输入服务启动时显示的访问令牌。</p><input id="token" type="password" autocomplete="off"><button id="login">进入</button>
<p id="error" role="alert"></p></main><script>document.getElementById('login').onclick=async()=>{let r=await fetch('/api/login',
{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:document.getElementById('token').value})});
if(r.ok)location.reload();else document.getElementById('error').textContent=(await r.json()).error}</script></html>"""


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default="127.0.0.1", help="使用 0.0.0.0 允许局域网访问")
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument("--data-dir", type=Path, default=data_directory())
  args = parser.parse_args()
  manager = RunManager(args.data_dir)
  server = http.server.ThreadingHTTPServer((args.host, args.port), DashboardHandler)
  server.manager = manager
  server.replay = ReplayRenderer(manager.root)
  server.token = os.environ.get("PIPER_H_DASHBOARD_TOKEN") or secrets.token_urlsafe(24)
  print(f"PiPER H 页面：http://{args.host}:{args.port}/", flush=True)
  print(f"访问令牌：{server.token}", flush=True)
  print(f"运行历史：{manager.root}", flush=True)
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    manager.stop()
    server.server_close()
    server.replay.close()


if __name__ == "__main__":
  main()

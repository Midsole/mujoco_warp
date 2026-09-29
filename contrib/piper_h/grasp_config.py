"""Validated settings for repeatable PiPER H grasp experiments."""

import math
import re

WORLD_COUNTS = (1, 16, 32, 64, 128, 256, 512, 1024)
MAX_WORLDS = max(WORLD_COUNTS)

FIELDS = [
  {
    "key": "sdf_mode",
    "label": "SDF 查询方式",
    "group": "机械臂与夹爪",
    "default": "dense",
    "choices": ["dense", "octree"],
    "choice_labels": {"dense": "稠密 SDF（默认，Warp GPU）", "octree": "八叉树 SDF（原生对照）"},
  },
  {
    "key": "object_shape",
    "label": "抓取物体",
    "group": "物体与接触",
    "default": "cube",
    "choices": ["cube", "cat", "uploaded"],
    "choice_labels": {"cube": "方块（网格 SDF）", "cat": "猫手机支架（网格 SDF）", "uploaded": "上传 STL（网格 SDF）"},
  },
  {
    "key": "finger_collision",
    "label": "手指碰撞模型",
    "group": "机械臂与夹爪",
    "default": "sdf",
    "choices": ["sdf", "box"],
    "choice_labels": {"sdf": "SDF（完整手指网格）", "box": "盒体（原始近似）"},
  },
  {
    "key": "nworld",
    "label": "并行仿真场景数",
    "group": "批量仿真",
    "default": 1,
    "min": 1,
    "max": MAX_WORLDS,
    "integer": True,
    "choices": list(WORLD_COUNTS),
  },
  {"key": "cat_mass", "label": "物体质量 (kg)", "group": "物体与接触", "default": 0.1, "min": 0.005, "max": 2},
  {
    "key": "condim",
    "label": "接触维数 condim",
    "group": "物体与接触",
    "default": 6,
    "min": 1,
    "max": 6,
    "integer": True,
    "choices": [1, 3, 4, 6],
  },
  {"key": "cat_friction", "label": "物体滑动摩擦", "group": "物体与接触", "default": 0.8, "min": 0, "max": 3},
  {"key": "cat_torsional_friction", "label": "物体扭转摩擦", "group": "物体与接触", "default": 0.005, "min": 0, "max": 3},
  {"key": "cat_rolling_friction", "label": "物体滚动摩擦", "group": "物体与接触", "default": 0.0001, "min": 0, "max": 3},
  {"key": "table_friction", "label": "桌面滑动摩擦", "group": "物体与接触", "default": 0.8, "min": 0, "max": 3},
  {"key": "table_torsional_friction", "label": "桌面扭转摩擦", "group": "物体与接触", "default": 0.005, "min": 0, "max": 3},
  {"key": "table_rolling_friction", "label": "桌面滚动摩擦", "group": "物体与接触", "default": 0.0001, "min": 0, "max": 3},
  {
    "key": "cat_contact_time",
    "label": "物体接触时间常数 (s)",
    "group": "物体与接触",
    "default": 0.004,
    "min": 0.001,
    "max": 0.1,
  },
  {
    "key": "table_contact_time",
    "label": "桌面接触时间常数 (s)",
    "group": "物体与接触",
    "default": 0.004,
    "min": 0.001,
    "max": 0.1,
  },
  {"key": "arm_kp", "label": "六轴 Kp", "group": "机械臂与夹爪", "default": [80, 140, 120, 35, 22, 12], "min": 0, "max": 1000},
  {"key": "arm_kv", "label": "六轴 Kv", "group": "机械臂与夹爪", "default": [5, 8, 7, 2.5, 1.8, 1.2], "min": 0, "max": 100},
  {
    "key": "arm_force_limit",
    "label": "六轴力矩上限 (N·m)",
    "group": "机械臂与夹爪",
    "default": [30, 30, 24, 12, 10, 8],
    "min": 0.1,
    "max": 200,
  },
  {"key": "arm_damping", "label": "六轴关节阻尼", "group": "机械臂与夹爪", "default": [0.12] * 6, "min": 0, "max": 10},
  {"key": "gripper_kp", "label": "夹爪 Kp", "group": "机械臂与夹爪", "default": 400, "min": 0, "max": 2000},
  {"key": "gripper_kv", "label": "夹爪 Kv", "group": "机械臂与夹爪", "default": 4, "min": 0, "max": 100},
  {"key": "gripper_force_limit", "label": "夹爪力上限 (N)", "group": "机械臂与夹爪", "default": 10, "min": 0.1, "max": 100},
  {"key": "gripper_damping", "label": "手指关节阻尼", "group": "机械臂与夹爪", "default": 0.1, "min": 0, "max": 10},
  {"key": "gravcomp", "label": "机器人重力补偿比例", "group": "机械臂与夹爪", "default": 1, "min": 0, "max": 1},
  {"key": "sdf_depth", "label": "SDF 八叉树深度", "group": "高级仿真", "default": 8, "min": 5, "max": 10, "integer": True},
  {
    "key": "sdf_initpoints",
    "label": "SDF 初始查询点",
    "group": "高级仿真",
    "default": 40,
    "min": 4,
    "max": 100,
    "integer": True,
  },
  {"key": "sdf_iterations", "label": "SDF 查询迭代", "group": "高级仿真", "default": 10, "min": 1, "max": 40, "integer": True},
  {"key": "timestep", "label": "物理步长 (s)", "group": "步进与任务", "default": 0.0005, "min": 0.0005, "max": 0.005},
  {"key": "control_hz", "label": "控制目标更新频率 (Hz)", "group": "步进与任务", "default": 2000, "min": 1, "max": 2000},
  {"key": "duration", "label": "任务时长 (s)", "group": "步进与任务", "default": 12, "min": 1, "max": 60},
  {
    "key": "solver_iterations",
    "label": "求解迭代次数",
    "group": "高级仿真",
    "default": 50,
    "min": 10,
    "max": 200,
    "integer": True,
  },
  {"key": "nconmax", "label": "每世界接触容量", "group": "高级仿真", "default": 256, "min": 64, "max": 1024, "integer": True},
  {"key": "njmax", "label": "每世界约束容量", "group": "高级仿真", "default": 1024, "min": 256, "max": 4096, "integer": True},
  {"key": "minimum_carried_height", "label": "携带最低高度 (m)", "group": "验收标准", "default": 0.08, "min": 0, "max": 0.3},
  {"key": "placement_tolerance", "label": "放置误差上限 (m)", "group": "验收标准", "default": 0.05, "min": 0, "max": 0.2},
  {"key": "minimum_contact_fraction", "label": "双指接触比例下限", "group": "验收标准", "default": 0.95, "min": 0, "max": 1},
  {"key": "maximum_final_drift", "label": "最终漂移上限 (m)", "group": "验收标准", "default": 0.005, "min": 0, "max": 0.1},
]

DEFAULTS = {**{field["key"]: field["default"] for field in FIELDS}, "object_id": ""}
ENGINES = ("c", "warp")


def validate_config(raw):
  """Return a full config, rejecting unknown or invalid fields."""
  if not isinstance(raw, dict):
    raise ValueError("配置必须是 JSON 对象")
  unknown = set(raw) - set(DEFAULTS) - {"engine"}
  if unknown:
    raise ValueError(f"未知参数：{', '.join(sorted(unknown))}")
  engine = raw.get("engine", "warp")
  if engine not in ENGINES:
    raise ValueError("engine 必须是 c 或 warp")
  object_id = raw.get("object_id", "")
  if not isinstance(object_id, str) or (object_id and not re.fullmatch(r"[0-9a-f]{32}", object_id)):
    raise ValueError("上传物体编号无效")
  result = {"engine": engine, "object_id": object_id}
  for field in FIELDS:
    key = field["key"]
    value = raw.get(key, field["default"])
    if isinstance(field["default"], str):
      if value not in field["choices"]:
        raise ValueError(f"{key} 只能是 {', '.join(field['choices'])}")
      result[key] = value
      continue
    items = value if isinstance(field["default"], list) else [value]
    if isinstance(field["default"], list) and (not isinstance(value, list) or len(value) != 6):
      raise ValueError(f"{key} 必须恰好有六个数值")
    checked = []
    for item in items:
      if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError(f"{key} 必须是有限数值")
      if field.get("integer") and (not isinstance(item, int) or isinstance(item, bool)):
        raise ValueError(f"{key} 必须是整数")
      if "choices" in field and item not in field["choices"]:
        raise ValueError(f"{key} 只能是 {', '.join(map(str, field['choices']))}")
      if not field["min"] <= item <= field["max"]:
        raise ValueError(f"{key} 必须在 {field['min']} 到 {field['max']} 之间")
      if not math.isfinite(item):
        raise ValueError(f"{key} 必须是有限数值")
      checked.append(item)
    result[key] = checked if isinstance(field["default"], list) else checked[0]
  if result["object_shape"] == "uploaded" and not object_id:
    raise ValueError("请先上传 STL 或选择已有上传物体")
  if engine == "c" and result["nworld"] != 1:
    raise ValueError("批量仿真需要 Warp GPU 后端；MuJoCo CPU 仅支持 1 个场景")
  if engine == "c":
    if raw.get("sdf_mode") == "dense":
      raise ValueError("稠密 SDF 需要 Warp GPU 后端；MuJoCo CPU 使用八叉树 SDF")
    result["sdf_mode"] = "octree"
  return result


def timing_settings(config):
  """Quantize control updates and task length to complete physical steps."""
  timestep = config["timestep"]
  control_steps = max(1, math.floor(1 / (config["control_hz"] * timestep) + 0.5))
  steps = max(1, math.floor(config["duration"] / timestep + 0.5))
  return {
    "physics_hz": 1 / timestep,
    "control_steps": control_steps,
    "effective_control_hz": 1 / (control_steps * timestep),
    "physics_steps": steps,
    "duration": steps * timestep,
  }


def history_config(raw):
  """Preserve the timing and SDF meaning of records predating these fields."""
  timestep = raw.get("timestep", 0.001)
  return {
    "finger_collision": "box",
    "object_shape": "cat",
    "timestep": timestep,
    "control_hz": 1 / timestep,
    "duration": 24,
    "sdf_mode": "octree",
    **raw,
  }

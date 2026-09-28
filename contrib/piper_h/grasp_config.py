"""Validated settings for repeatable PiPER H grasp experiments."""

import math

FIELDS = [
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
  {"key": "timestep", "label": "物理步长 (s)", "group": "高级仿真", "default": 0.001, "min": 0.0005, "max": 0.005},
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

DEFAULTS = {field["key"]: field["default"] for field in FIELDS}
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
  result = {"engine": engine}
  for field in FIELDS:
    key = field["key"]
    value = raw.get(key, field["default"])
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
  return result

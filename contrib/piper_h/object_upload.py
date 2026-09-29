"""Store closed STL objects as immutable, metre-scale meshes for grasp experiments."""

import hashlib
import json
import re
import tempfile
import uuid
from pathlib import Path

import numpy as np
from prepare_cat import STL_RECORD

MAX_UPLOAD = 64 * 1024 * 1024
MAX_TRIANGLES = 1_000_000
OBJECT_ID = re.compile(r"[0-9a-f]{32}\Z")
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_VECTOR = rf"({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})"
_FACET = re.compile(
  rf"facet\s+normal\s+{_NUMBER}\s+{_NUMBER}\s+{_NUMBER}\s+outer\s+loop\s+"
  rf"vertex\s+{_VECTOR}\s+vertex\s+{_VECTOR}\s+vertex\s+{_VECTOR}\s+endloop\s+endfacet",
  re.IGNORECASE,
)


def read_triangles(raw):
  """Decode binary or ASCII STL, including binary headers beginning with 'solid'."""
  if not raw or len(raw) > MAX_UPLOAD:
    raise ValueError("STL 文件必须非空且不超过 64 MiB")
  if len(raw) >= 84:
    count = int.from_bytes(raw[80:84], "little")
    if count and len(raw) == 84 + count * 50:
      if count > MAX_TRIANGLES:
        raise ValueError("STL 三角形数量不能超过 1,000,000")
      return np.frombuffer(raw, dtype=STL_RECORD, offset=84, count=count)["vertices"].astype(np.float64)
  try:
    text = raw.decode("ascii").strip()
  except UnicodeDecodeError as exc:
    raise ValueError("STL 格式无效或二进制三角形数据不完整") from exc
  lines = text.splitlines()
  if len(lines) < 3 or not re.match(r"solid(?:\s|$)", lines[0], re.I) or not re.match(r"endsolid(?:\s|$)", lines[-1], re.I):
    raise ValueError("需要完整的二进制或 ASCII STL 文件")
  body = "\n".join(lines[1:-1])
  triangles, end = [], 0
  for match in _FACET.finditer(body):
    if body[end : match.start()].strip():
      raise ValueError("ASCII STL 三角形格式无效")
    triangles.append(match.groups())
    end = match.end()
    if len(triangles) > MAX_TRIANGLES:
      raise ValueError("STL 三角形数量不能超过 1,000,000")
  if not triangles or body[end:].strip():
    raise ValueError("ASCII STL 三角形格式无效或文件不完整")
  return np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)


def prepare_mesh(raw, unit, size_mm):
  """Preserve all triangles, validate a closed oriented surface, and place its bottom at Z=0."""
  if unit not in ("mm", "m"):
    raise ValueError("STL 单位只能是 mm 或 m")
  if not np.isfinite(size_mm) or not 0 <= size_mm <= 150:
    raise ValueError("最长边设置须为 0–150 mm；0 表示保留原始尺寸")
  triangles = read_triangles(raw)
  if not np.isfinite(triangles).all():
    raise ValueError("STL 包含非有限坐标")
  vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
  faces = inverse.reshape(-1, 3)
  lower, upper = vertices.min(axis=0), vertices.max(axis=0)
  extents = (upper - lower) * (0.001 if unit == "mm" else 1)
  if not np.isfinite(extents).all() or np.any(extents <= 0):
    raise ValueError("STL 必须是具有三维体积的物体")
  scale = (0.001 if unit == "mm" else 1) * (size_mm * 0.001 / extents.max() if size_mm else 1)
  offset = np.r_[lower[:2] + (upper[:2] - lower[:2]) / 2, lower[2]]
  vertices = (vertices - offset) * scale
  points = vertices[faces]
  if np.any(np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1) == 0):
    raise ValueError("STL 包含退化三角形，请先修复网格")
  edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
  _, indices, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
  direction = np.where(edges[:, 0] < edges[:, 1], 1, -1)
  if np.any(counts != 2) or np.any(np.bincount(indices, weights=direction) != 0):
    raise ValueError("SDF 需要封闭且面方向一致的 STL；请修复孔洞、非流形边或翻转面")
  volume = np.einsum("ij,ij->i", points[:, 0], np.cross(points[:, 1], points[:, 2])).sum() / 6
  if abs(volume) <= np.prod(np.ptp(vertices, axis=0)) * 1e-10:
    raise ValueError("STL 封闭体积无效")
  if volume < 0:
    faces = faces[:, [0, 2, 1]]
  bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
  dimensions = bounds[1] - bounds[0]
  if dimensions[0] > 0.085 or dimensions.max() > 0.15 or dimensions.min() < 0.002:
    raise ValueError("当前抓取场景支持 X 宽度 ≤85 mm、最长边 ≤150 mm、各轴尺寸 ≥2 mm；可调整最长边后上传")
  return (
    vertices,
    faces,
    {
      "unit": unit,
      "size_mm": size_mm,
      "source_dimensions_mm": (extents * 1000).tolist(),
      "dimensions_mm": (dimensions * 1000).tolist(),
      "bounds_m": bounds.tolist(),
      "source_offset": offset.tolist(),
      "scale": scale,
      "triangles": len(faces),
      "vertices": len(vertices),
      "source_sha256": hashlib.sha256(raw).hexdigest(),
    },
  )


class ObjectStore:
  """Keep uploaded assets outside the source checkout and publish complete uploads atomically."""

  def __init__(self, root):
    self.root = Path(root) / "objects"
    self.root.mkdir(parents=True, exist_ok=True)

  def get(self, object_id):
    if not isinstance(object_id, str) or not OBJECT_ID.fullmatch(object_id):
      raise ValueError("上传物体编号无效")
    directory = self.root / object_id
    if not (directory / "metadata.json").exists() or not (directory / "object.obj").exists():
      raise ValueError("上传物体不存在，请重新上传或选择已有物体")
    return json.loads((directory / "metadata.json").read_text())

  def list_objects(self):
    return [self.get(path.name) for path in sorted(self.root.iterdir()) if OBJECT_ID.fullmatch(path.name)]

  def upload(self, raw, filename, unit="mm", size_mm=60.0):
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not name.lower().endswith(".stl") or len(name) > 200:
      raise ValueError("请选择名称不超过 200 字符的 .stl 文件")
    vertices, faces, metadata = prepare_mesh(raw, unit, size_mm)
    object_id = uuid.uuid4().hex
    metadata.update(id=object_id, name=name)
    with tempfile.TemporaryDirectory(dir=self.root, prefix=".upload-") as temporary:
      directory = Path(temporary)
      (directory / "source.stl").write_bytes(raw)
      with (directory / "object.obj").open("w") as stream:
        stream.write("# Uploaded STL; metres; XY centre; bottom Z=0.\n")
        np.savetxt(stream, vertices, fmt="v %.17g %.17g %.17g")
        np.savetxt(stream, faces + 1, fmt="f %d %d %d")
      metadata["obj_sha256"] = hashlib.sha256((directory / "object.obj").read_bytes()).hexdigest()
      (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
      directory.rename(self.root / object_id)
    return metadata

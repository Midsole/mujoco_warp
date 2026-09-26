"""Convert the original binary STL to a full-resolution, metre-scale OBJ."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DIRECTORY = Path(__file__).resolve().parent
STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def read_stl(source: Path):
  raw = source.read_bytes()
  if len(raw) < 84:
    raise ValueError("Truncated binary STL")
  count = int.from_bytes(raw[80:84], "little")
  if not count or len(raw) != 84 + 50 * count:
    raise ValueError("Expected a binary STL with a complete triangle array")
  records = np.frombuffer(raw, dtype=STL_RECORD, offset=84, count=count)
  vertices, inverse = np.unique(records["vertices"].reshape(-1, 3), axis=0, return_inverse=True)
  if not np.isfinite(vertices).all():
    raise ValueError("STL contains non-finite coordinates")
  return raw, vertices.astype(np.float64), inverse.reshape(-1, 3)


def prepare(source: Path, directory: Path = DIRECTORY):
  raw, vertices, faces = read_stl(source)
  lower, upper = vertices.min(axis=0), vertices.max(axis=0)
  edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
  _, counts = np.unique(edges, axis=0, return_counts=True)
  if np.any(counts != 2):
    raise ValueError("SDF source must be closed: found edges not shared by exactly two triangles")
  offset = np.r_[(lower[:2] + upper[:2]) / 2, lower[2]]
  vertices = (vertices - offset) * 0.001
  target = directory / "meshes" / "cat_phone_stand.obj"
  target.parent.mkdir(parents=True, exist_ok=True)
  with target.open("w") as stream:
    stream.write("# Full original triangulation; metres; XY bounding-box centre, bottom Z=0.\n")
    np.savetxt(stream, vertices, fmt="v %.17g %.17g %.17g")
    np.savetxt(stream, faces + 1, fmt="f %d %d %d")
  metadata = {
    "source": str(source.resolve()),
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "obj_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "triangles": len(faces),
    "vertices": len(vertices),
    "source_bounds_mm": [lower.tolist(), upper.tolist()],
    "source_offset_mm": offset.tolist(),
    "scale": 0.001,
    "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
    "nonmanifold_edges": 0,
    "notes": "No decimation or convex decomposition; duplicate STL vertices are merged without changing triangles.",
  }
  (directory / "cat_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
  return metadata


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, required=True, help="original binary STL, with coordinates in millimetres")
  args = parser.parse_args()
  print(json.dumps(prepare(args.source), indent=2))


if __name__ == "__main__":
  main()

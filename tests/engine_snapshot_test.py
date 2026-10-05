"""Ensure the submodule split preserves the original engine exactly."""

import hashlib
import json
from pathlib import Path

import mujoco_warp

ROOT = Path(__file__).resolve().parents[1]


def test_loaded_engine_matches_original_snapshot():
  engine = ROOT / ".build/mujoco_warp"
  assert Path(mujoco_warp.__file__).resolve().parent == engine / "mujoco_warp"
  snapshot = json.loads((ROOT / "patches/mujoco_warp/source-snapshot.json").read_text())
  for name, expected in snapshot.items():
    assert hashlib.sha256((engine / name).read_bytes()).hexdigest() == expected, name

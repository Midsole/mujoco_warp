"""Keep shared-device GPU suites together when using pytest-xdist."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
  """Avoid concurrent engine/PiPER CUDA graph allocations on the same GPU."""
  suites = (ROOT / ".build/mujoco_warp/mujoco_warp", ROOT / "contrib/piper_h")
  for item in items:
    if any(Path(item.path).is_relative_to(suite) for suite in suites):
      item.add_marker(pytest.mark.xdist_group("shared_cuda"))

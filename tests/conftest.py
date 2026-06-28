"""Shared test config: repo root on sys.path, JAX forced to CPU."""
import os
import sys
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))          # so `import nff.sofa...` works
sys.path.insert(0, str(REPO / "sofa")) # so the oracle-side `materials` imports flat

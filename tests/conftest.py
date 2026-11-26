"""Test-session setup.

The default jax-metal backend fails test collection (XlaRuntimeError: UNIMPLEMENTED), so the suite
runs on CPU unless the caller explicitly chooses another platform. This must run before any test
module imports JAX, which is why it lives here.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

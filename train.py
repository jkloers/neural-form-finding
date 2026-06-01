"""
train.py — Neural Form-Finding entry point.

Usage:
  python train.py --config-name <name>
  python train.py --arch architectures/mpnn_base --suite problems/suite_2x2_rdqk
  python train.py --arch architectures/mpnn_base --suite problems/suite_2x2_rdqk \
                  --problem-ids p001,p002

All logic lives in nff/scripts/train.py.
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

from nff.scripts.train import main

if __name__ == "__main__":
    main()

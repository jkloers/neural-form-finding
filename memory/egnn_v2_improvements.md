---
name: egnn-v2-improvements
description: EGNN hyperparameter improvements applied 2026-05-20 to close performance gap vs polynomial mappings
metadata:
  type: project
---

Implemented targeted improvements to close GNN vs polynomial performance gap. Goal: visually perfect tessellation with closed hinges, no intersections.

**Changes made:**

1. `egnn.py` — `phi_x` init scale: `0.01 → 0.05` (5× larger initial coord movement = better gradient signal on position pathway)
2. `egnn.py` — rotation range: `±π/2 → ±π` (flat→circle can require >90° face rotation)
3. `config.py` + `trainer.py` — added `lr_schedule: "cosine"` option (cosine decay to 10% of init lr over num_epochs)
4. New config: `data/configs/gnn/poster1_egnn_v2.yaml` — run with `python train.py --config-dir gnn --config-name poster1_egnn_v2`

**Config v2 key values vs v1:**
- `hidden_dim`: 32 → 64
- `num_layers`: 3 → 4
- `grad_clip`: 0.1 → 1.0 (critical: clip=0.1 limited total displacement to ~0.3 over 300 epochs)
- `hinge_gap`: 100 → 500 (ratio vs chamfer: 1:20 → 1:4, stronger connectivity signal)
- `anchoring`: 2000 → 3000
- `regularization`: 0.001 → 0.0001
- `num_epochs`: 300 → 500
- `lr_schedule`: cosine

**Why:** Polynomial needs ~15 params; GNN has ~80k with hidden=64. GNN must discover what polynomial encodes analytically, so needs more epochs, more freedom (less clipping), and stronger connectivity incentive.

**How to apply:** Use `poster1_egnn_v2.yaml` for all new GNN experiments. If hinges still open after v2, consider two-phase curriculum (phase 1: hinge_gap=2000 for 150 epochs, phase 2: normal).

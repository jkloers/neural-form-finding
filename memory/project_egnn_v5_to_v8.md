---
name: egnn-v5-to-v8
description: EGNN v5–v8 experiments 2026-05-20: from open hinges to closed hinges, validity solver tuning results
metadata:
  type: project
---

**Achievement:** Closed hinges in Stage 1 and Stage 2, previously open in v3_s2.

**Best configs (run with `python train.py --config-dir gnn --config-name <name>`):**
- `poster1_egnn_v8` (seed=2): Square fill, closed hinges, best area: Chamfer 0.036, ΔArea -0.28
- `poster1_egnn_v8_s3` (seed=3): Diamond RDQK_D pattern, closed hinges: Chamfer 0.038, ΔArea -1.02

**Key findings:**

1. **Validity solver weights that work**: `connectivity=3000, anchoring=300, void_length=500, void_collinear=500, target=200, face_area=100`. The ratio connectivity >> anchoring (10:1) is what enables hinge closure. Previous configs had anchoring >> connectivity (20:1 reverse), preventing any hinge closure in Stage 1.

2. **face_area=100 in validity**: Forces Stage 1 to close hinges via centroid translation + rotation ONLY (rigid-body kinematics). Without this (v7 face_area=5), Stage 1 resizes faces to satisfy constraints → ΔArea=-1.03. With it, square-orientation configs get ΔArea=-0.25.

3. **Two local minima**: The GNN with v8 settings finds either a "diamond" orientation (loss≈191, ΔArea≈-1.0, more visually correct RDQK_D) or a "square" orientation (loss≈92, ΔArea≈-0.25, better area). Seed determines which minimum is found.

4. **target=200 in validity**: Makes Stage 1 also fit the circle boundary (Chamfer distance). Important for chamfer quality when anchoring is low. v5 (no target) had chamfer 0.031 vs v7/v8 with target=200 having 0.036-0.038 (better Stage 1 closure quality but GNN learns differently).

5. **Chamfer gap**: Current best (0.036) vs v3_s2 (0.020). Gap caused by Stage 1 rearranging faces from Stage 0 (IFT gradient disconnects Stage 0 from chamfer signal). v3_s2 had anchoring=2000 → Stage 1 ≈ identity → Stage 0 directly learned chamfer.

**Remaining issues:**
- Face 0 (clamped) always outside the target circle — geometric constraint (face 0 is corner of flat grid; hinge points between face 0 and neighbors are at circle boundary, so face 0 centroid must be outside).
- Chamfer gap: 0.036 vs 0.020 (v3_s2). Would require curriculum training or architectural changes to fix.

**What to try next:**
1. Curriculum training: Phase 1 (0-200 ep): anchoring=2000, connectivity=100 (Stage 0 learns circle); Phase 2 (200-500): connectivity=3000, anchoring=300 (hinge closure)
2. Per-node GNN corrections: add output head predicting δ_{f,n} per face node
3. Face 0 loss: explicit penalty for face 0 centroid outside circle

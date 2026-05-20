# Memory Index

- [GNN Foundations](project_gnn_foundations.md) — graph builder, dummy GNN, gradient validated, JAX-Metal JIT bug workaround
- [EGNN v2 Improvements](egnn_v2_improvements.md) — param tuning 2026-05-20: rotation ±π, phi_x init 0.05, grad_clip 1.0, cosine LR, hidden=64 layers=4, hinge_gap=500
- [Jacobian CNV Transform](project_jacobian_approach.md) — full Jacobian beats independent scale/theta; det-normalization (v4) breaks chamfer; v5+ uses full F
- [Validity Solver Diagnosis](project_validity_solver.md) — anchoring=2000 >> connectivity=100 → hinges never closed; fix: connectivity=3000, anchoring=300, void=500
- [EGNN v5–v8 Results](project_egnn_v5_to_v8.md) — closed hinges achieved; best: v8/v8_s3 with target=200, face_area=100; chamfer gap 0.036 vs 0.020 remains

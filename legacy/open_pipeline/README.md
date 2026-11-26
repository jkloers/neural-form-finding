# legacy/open_pipeline/

The open/deployed RDQK pattern needed a Stage-1 validity solver after every mapping. These files are
the parts of it that the live pipeline no longer uses:

- `validity_lbfgs.py` — the L-BFGS Stage-1 solver (jaxopt, differentiable via custom_vjp), which
  minimised the weighted penalties in `constraints.py`.
- `constraints.py` — the full set of open-pattern validity penalties (hinge non-intersection, arm
  symmetry, void parallelograms, anchoring, ...). Only `hinge_connectivity` is still live, in
  `nff/stages/constraints.py`.
- `mapping_with_open_maps.py` — `nff/stages/mapping.py` as it was, with the analytic open-pattern maps
  (conformal polynomial, asymmetric roots, elliptical grip, homothetic, boundary projection) and
  `direct_transform`.
- `notebooks/` — exploration of those maps.

Still live in `nff/`: the free-vertex map (`direct_vertices`) and Stage-1 alternating projections
(`nff/stages/projection.py`), because the chart ablation (`nff/analysis/chart_ablation.py`) that
measures what the closed chart buys runs its free-vertex arms through the live pipeline.

Run any of this from tag `pre-cleanup-2026-11`.

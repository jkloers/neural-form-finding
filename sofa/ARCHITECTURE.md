# SOFA oracle — code architecture

The SOFA integration is split across **two environments by design**. Keeping the
boundary clean is what lets a slow, runtime-hostile FEM solver behave like a
differentiable function the JAX pipeline can call.

## Two worlds

| | **Oracle side** — `sofa/` | **Client side** — `nff/sofa/` |
|---|---|---|
| Runs in | Docker image / local SOFA runtime | `kgnn_mac` (the JAX env) |
| Imports `Sofa.*` | **yes** | **never** |
| Reaches the oracle | *is* the oracle | over HTTP via the `tesseract_core` SDK |

The two never share a process. The client speaks to the oracle only through
`/apply` and `/jacobian` (see `tesseract_api.py`).

```
   nff/sofa/  (client, no Sofa)                 sofa/  (oracle, imports Sofa)
   ─────────────────────────────                ─────────────────────────────
   hinge_optimizer.py ──tesseract_core SDK──▶ Docker ▶ tesseract_api.py
   oracle_payload.py     (HTTP /apply,/jacobian)  │ imports
   mesh_builder_gmsh.py  ◀──symlink─────────────  │ simulate_cell.py
   hinge_geometry.py     ◀──symlink─────────────  │   ├─ scene_builder.py
   fatigue.py / hinge_viz.py                      │   └─ materials.py
```
`nff/sofa/hinge_optimizer.py` is the optimization driver: client code that runs in
`kgnn_mac` and imports only from `nff.sofa.*`, never `Sofa`.

## File responsibilities

**Oracle side (`sofa/`)** — packaged into the Tesseract Docker image:
- `tesseract_api.py` — InputSchema / OutputSchema + `apply()` / `jacobian()` (FD).
- `simulate_cell.py` — `evaluate_unit_cell()`; the only file that imports `Sofa`.
- `scene_builder.py` — builds the SOFA scene (BCs, rotation/shear/tension loading).
- `materials.py` — SvK energy, von Mises, principal strain, KS smooth-max aggregate.
- `mesh_builder_gmsh.py` — **symlink** to `nff/sofa/mesh_builder_gmsh.py`: `build_mesh_gmsh` (the mesher).
- `hinge_geometry.py` — **symlink** to `nff/sofa/hinge_geometry.py`: `compute_hinge_geometry` (geometry resolver, shared with the mesher).

**Client side (`nff/sofa/`)** — importable from `kgnn_mac`, never touches `Sofa`:
- `hinge_optimizer.py` — the optimization driver (CLI entry point): loss, NumPy Adam, history.
- `oracle_payload.py` — builds the oracle input dict (`build_physical_cs`, `build_payload`,
  `PARAM_NAMES`). Transport is the `tesseract_core` SDK (`Tesseract.from_url(url).apply()
  /.jacobian()`), which also decodes responses — no hand-rolled HTTP/JSON layer.
- `mesh_builder_gmsh.py` — **source of truth** for `build_mesh_gmsh` (`CentroidalState → gmsh tet mesh`).
- `hinge_geometry.py` — **source of truth** for `compute_hinge_geometry` (faces pushed apart + Bézier arcs); shared by the mesher, the optimizer (analytic hinge area), and viz.
- `fatigue.py` — Coffin-Manson `cycles_to_failure`.
- `hinge_viz.py` — Princeton palette + shared mesh/Bézier plotting helpers.

`scripts/` holds single-purpose figure/CLI tools (plus `config_to_physical.py`, a
legacy parametric-scale helper, no longer imported) that pull shared logic from
`nff.sofa.*` — no duplicated payloads, fatigue, or palette.

## The shared-file symlinks

Two files are shared with the oracle via symlink:
`sofa/mesh_builder_gmsh.py → ../nff/sofa/mesh_builder_gmsh.py` and
`sofa/hinge_geometry.py → ../nff/sofa/hinge_geometry.py`. Each is a single source of
truth, so there is **no manual copy to keep in sync**. Edit the files under
`nff/sofa/`. If a future `tesseract build sofa/` ever fails to follow a symlink into
the Docker context, replace it with a real copy and add an identity check.

## End-to-end flow

The build runs once; the epoch loop is the heart. The `tesseract_core` SDK is the
**membrane** — everything left of it is optimization (never touches SOFA),
everything right of it is physics (never touches the optimizer); `oracle_payload`
builds the request dict the SDK sends. The image itself is built by the external CLI
`tesseract build sofa/` reading `tesseract_config.yaml` (no build code lives in the repo).

```
PHASE 0 — BUILD  (once, needs Docker)
   tesseract build sofa/   reads tesseract_config.yaml (+ requirements) → Docker image
   docker run … serve      → oracle on localhost:8000

PHASE 1 — SETUP  (once per run)                         [CLIENT, kgnn_mac]
   hinge_optimizer.main → op.build_physical_cs   (patterns.yaml → CentroidalState)
                        → Tesseract.from_url(url) (the tesseract_core SDK handle)
                        → _initial_params         (hinge_geometry.compute_hinge_geometry)

PHASE 2 — EPOCH LOOP  (×n_epochs)
   CLIENT (kgnn_mac)                       │  SERVER (Docker)
   ────────────────────────────────       │  ──────────────────────────────────────
   ① payload = op.build_payload           │
   ② fwd = oracle.apply ──SDK /apply──────▶ tesseract_api.apply
                                           │   ├ build_mesh_gmsh → hinge_geometry
                                           │   └ simulate_cell → scene_builder
                                           │       → SOFA solve → materials
             value ◀──────(SDK decodes)────  OutputSchema
   ③ jac = oracle.jacobian ─SDK /jacobian─▶ tesseract_api.jacobian
                                           │   └ finite_difference_jacobian
                                           │       → apply() ×18 (nudge each knob ±ε)
          gradient ◀──────(SDK decodes)────  {strain: {9 knobs: ∂}}
   ④ _hinge_area (hinge_geometry, LOCAL); fatigue.cycles_to_failure
     grad = d_fat(from jac) + d_mat + d_gap
   ⑤ _NumpyAdam.update → project floor → new params ── loop back to ① ──┘

PHASE 3 — CAPTURE + VIZ  (after the loop)               [CLIENT]
   _capture_final_state → oracle.apply(return_fields=True) → final_state.npz
   scripts/visualize_*.py → build_mesh_gmsh + hinge_geometry + hinge_viz → PNGs
```

Per epoch the client asks the oracle **twice**: `/apply` for the value ("how
good?") and `/jacobian` for the gradient ("which way to improve?", computed by
re-running `apply` with each knob perturbed). Only the strain term costs SOFA; the
optimizer adds the cheap analytic gradients (area, gap) locally and takes one Adam
step.

# SOFA oracle — code architecture

The SOFA integration is split across **two environments by design**. Keeping the
boundary clean is what lets a slow, runtime-hostile FEM solver behave like a
differentiable function the JAX pipeline can call.

## Two worlds

| | **Oracle side** — `sofa/` | **Client side** — `nff/sofa/` |
|---|---|---|
| Runs in | Docker image / local SOFA runtime | `kgnn_mac` (the JAX env) |
| Imports `Sofa.*` | **yes** | **never** |
| Reaches the oracle | *is* the oracle | over HTTP (`tesseract_client`) |

The two never share a process. The client speaks to the oracle only through
`/apply` and `/jacobian` (see `tesseract_api.py`).

```
   nff/sofa/  (client, no Sofa)                 sofa/  (oracle, imports Sofa)
   ─────────────────────────────                ─────────────────────────────
   tesseract_client.py  ──HTTP──▶  Docker ▶  tesseract_api.py
   hinge_optimizer.py                (Tesseract)   │ imports
   mesh_builder_gmsh.py  ◀──symlink──────────────  │ simulate_cell.py
   fatigue.py / hinge_viz.py                       │   ├─ scene_builder.py
   config_to_physical.py                           │   └─ materials.py
```
`nff/sofa/hinge_optimizer.py` is the optimization driver: client code that runs in
`kgnn_mac` and imports only from `nff.sofa.*`, never `Sofa`.

## File responsibilities

**Oracle side (`sofa/`)** — packaged into the Tesseract Docker image:
- `tesseract_api.py` — InputSchema / OutputSchema + `apply()` / `jacobian()` (FD).
- `simulate_cell.py` — `evaluate_unit_cell()`; the only file that imports `Sofa`.
- `scene_builder.py` — builds the SOFA scene (BCs, rotation/shear/tension loading).
- `materials.py` — SvK energy, von Mises, principal strain, KS smooth-max aggregate.
- `mesh_builder_gmsh.py` — **symlink** to `nff/sofa/mesh_builder_gmsh.py` (one source).

**Client side (`nff/sofa/`)** — importable from `kgnn_mac`, never touches `Sofa`:
- `hinge_optimizer.py` — the optimization driver (CLI entry point): loss, NumPy Adam, history.
- `tesseract_client.py` — the single HTTP client: `apply` / `jacobian`,
  `decode_scalar` / `decode_array`, `build_physical_cs`, `build_payload`, `PARAM_NAMES`.
- `mesh_builder_gmsh.py` — **source of truth** for `CentroidalState → gmsh tet mesh`.
- `fatigue.py` — Coffin-Manson `cycles_to_failure`.
- `hinge_viz.py` — Princeton palette + shared mesh/Bézier plotting helpers.
- `config_to_physical.py` — YAML → physical CentroidalState namespace.

`scripts/` (hinge_optimizer driver aside) are thin, single-purpose figure/CLI tools
that import everything shared from `nff.sofa.*` — no duplicated decoders, payloads,
fatigue, or palette.

## The mesh-builder symlink

`sofa/mesh_builder_gmsh.py → ../nff/sofa/mesh_builder_gmsh.py` — a single source of
truth, so there is **no manual copy to keep in sync**. Edit the file under
`nff/sofa/`. If a future `tesseract build sofa/` ever fails to follow the symlink
into the Docker context, replace it with a real copy and add an identity check.

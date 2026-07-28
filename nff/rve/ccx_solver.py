"""Single-hinge deployment via CalculiX — use the solver's built-in mechanics.

We do NOT hand-code constitutive models. CalculiX owns the physics: finite-strain
(NLGEOM), von-Mises plasticity (*PLASTIC), automatic increment/cut-back continuation,
and rigid-body kinematics. We only (1) build the mesh, (2) write a small text deck,
(3) run ``ccx``, and (4) read energy / out-of-plane displacement / strain from the output.

Kinematics: clamp the left face-arc; on the right face-arc prescribe the rigid rotation
of each node about the pivot (inside the ligament). A small out-of-plane crest is baked
into the mesh so the ligament buckles the right way. Faces stay coplanar (z=0 on the arcs).

Mesh: quadratic 15-node wedges (C3D15) — thin extruded sheets need prisms; split tets
invert. Runs in the ``ccx`` conda env (gmsh + shapely + calculix).

Known limitation (validation-phase TODO): the rotation is applied in a single step, so the
displacement ramp is a straight line rather than the true arc — fine for small angles, to be
replaced by a multi-step (cumulative) arc path before large-angle production runs.
"""

import os
import re
import subprocess
import time

import numpy as np

from nff.rve.geometry import RVEParams, build_rve_domain, boundary_tag
from nff.rve.damage import (LIG_CENTER_FRAC, LIG_RADIUS_FRAC, max_principal_strain,
                            mean_triaxiality, peak_peeq)
from nff.rve.materials import Hypotheses, coerce_material
from nff.rve.materials.steel import STEEL   # re-exported for back-compat (callers import from here)

# CalculiX lives in its own conda env here (`/opt/miniconda3/envs/ccx/bin/ccx`, v2.23) and is NOT
# on the kgnn_mac PATH, so a bare "ccx" resolves only when the caller has arranged it. Override
# with $CCX_BIN or the ``ccx_bin`` argument.
CCX_BIN = os.environ.get("CCX_BIN", "ccx")


def _build_mesh(p: RVEParams, pivot, imp_amp, n_through, lc_min, lc_max):
    """Second-order (C3D15 prism) RVE mesh. Returns nodes, connectivity, arc node-id sets."""
    import gmsh
    dom = build_rve_domain(p)
    coords = list(dom.exterior.coords)[:-1]
    n = len(coords)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("rve")
        pts = [gmsh.model.geo.addPoint(x, y, 0.0, lc_max) for (x, y) in coords]
        lines, slit = [], []
        for i in range(n):
            a, b = coords[i], coords[(i + 1) % n]
            ln = gmsh.model.geo.addLine(pts[i], pts[(i + 1) % n]); lines.append(ln)
            mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
            if boundary_tag(mx, my, p) == "free" and my < -1e-6:
                slit.append(ln)
        surf = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])
        ext = gmsh.model.geo.extrude([(2, surf)], 0, 0, p.thickness,
                                     numElements=[n_through], recombine=True)   # -> prisms
        gmsh.model.geo.synchronize()
        vol = next(e for e in ext if e[0] == 3)
        # refine ONLY the uncut ligament strip (fillet-top -> secondary cut) -- the sole
        # deforming region. The free main cut below/beside the fillet stays coarse.
        gmsh.model.mesh.field.add("Ball", 1)
        gmsh.model.mesh.field.setNumber(1, "XCenter", 0.0)
        # same disc the damage average uses -- refined region == averaged region, one constant pair
        gmsh.model.mesh.field.setNumber(1, "YCenter", LIG_CENTER_FRAC * p.w_lig)   # mid-ligament
        gmsh.model.mesh.field.setNumber(1, "ZCenter", 0.5 * p.thickness)
        gmsh.model.mesh.field.setNumber(1, "Radius", LIG_RADIUS_FRAC * p.w_lig)    # fillet + strip + secondary
        gmsh.model.mesh.field.setNumber(1, "Thickness", LIG_RADIUS_FRAC * p.w_lig)  # transition to coarse
        gmsh.model.mesh.field.setNumber(1, "VIn", lc_min)
        gmsh.model.mesh.field.setNumber(1, "VOut", lc_max)
        gmsh.model.mesh.field.setAsBackgroundMesh(1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        # tag arcs by bounding box on the volume boundary side-surfaces
        arc = {"rigid_A": [], "rigid_B": []}
        for (d, s) in gmsh.model.getBoundary([vol], oriented=False):
            xmn, ymn, zmn, xmx, ymx, zmx = gmsh.model.getBoundingBox(2, s)
            if (zmx - zmn) < 1e-6:
                continue
            t = boundary_tag(0.5 * (xmn + xmx), 0.5 * (ymn + ymx), p)
            if t in arc:
                arc[t].append(s)
        for name, surfs in arc.items():
            gmsh.model.addPhysicalGroup(2, surfs, {"rigid_A": 1, "rigid_B": 2}[name])
        gmsh.model.addPhysicalGroup(3, [vol[1]], 1)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)    # 15-node prisms (C3D15)
        gmsh.model.mesh.setOrder(2)

        tags, xyz, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(xyz).reshape(-1, 3)
        remap = {int(t): i + 1 for i, t in enumerate(tags)}       # gmsh tag -> 1-based id
        # 15-node prisms (gmsh type 18); reorder to CalculiX C3D15
        g2c = [0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11]
        conn = []
        etypes, _, enodes = gmsh.model.mesh.getElements(dim=3)
        for et, en in zip(etypes, enodes):
            if int(et) == 18:                                     # 15-node prism (C3D15)
                arr = np.asarray(en, int).reshape(-1, 15)
                conn = [[remap[int(row[i])] for i in g2c] for row in arr]
        if not conn:
            raise RuntimeError("no C3D15 prisms extracted (check element order/type)")
        # fix orientation: CalculiX needs the bottom triangle CW seen from the top
        for k, row in enumerate(conn):
            v = xyz[[row[0] - 1, row[1] - 1, row[2] - 1, row[3] - 1, row[4] - 1, row[5] - 1]]
            n = np.cross(v[1] - v[0], v[2] - v[0])
            h = v[3:6].mean(0) - v[0:3].mean(0)
            if np.dot(n, h) < 0:                                  # top/bottom swapped
                conn[k] = [row[i] for i in [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 12, 13, 14]]
        # out-of-plane crest imperfection along x=0 over the ligament (after orientation fix)
        Py = pivot[1]; wx, wy = 2.0 * abs(Py), 1.4 * abs(Py)
        crest = np.exp(-(xyz[:, 0] / wx) ** 2) * np.exp(-((xyz[:, 1] - Py) / wy) ** 2)
        xyz[:, 2] += imp_amp * crest
        arcA = sorted({remap[int(t)] for t in gmsh.model.mesh.getNodesForPhysicalGroup(2, 1)[0]})
        arcB = sorted({remap[int(t)] for t in gmsh.model.mesh.getNodesForPhysicalGroup(2, 2)[0]})
        return xyz, conn, arcA, arcB
    finally:
        gmsh.finalize()


def _write_inp(path, xyz, conn, arcA, arcB, pivot, mat, angle, elastic_only, n_frames, hyp):
    L = ["*NODE"]
    for i, (x, y, z) in enumerate(xyz, start=1):
        L.append(f"{i}, {x:.6e}, {y:.6e}, {z:.6e}")
    L.append("*ELEMENT, TYPE=C3D15, ELSET=EALL")
    for e, row in enumerate(conn, start=1):
        L.append(f"{e}, " + ", ".join(str(v) for v in row))
    L.append("*NSET, NSET=ARCA\n" + ",\n".join(str(v) for v in arcA))
    L.append(mat.constitutive_cards(hyp, elastic_only=elastic_only))
    L.append(mat.section_cards("EALL", hyp))
    L.append("*STEP, NLGEOM, INC=1000")
    L.append("*STATIC")
    L.append(f"{1.0/n_frames:.4f}, 1.0, 1e-6, {1.0/n_frames:.4f}")
    L.append("*BOUNDARY")
    L.append("ARCA, 1, 3, 0.0")                                  # clamp the left face
    # right face: prescribe the rigid rotation of each arc node about the pivot (z=0)
    c, s = np.cos(angle), np.sin(angle)
    Px, Py = pivot
    for nid in arcB:
        x, y = xyz[nid - 1, 0], xyz[nid - 1, 1]
        ux = Px + c * (x - Px) - s * (y - Py) - x
        uy = Py + s * (x - Px) + c * (y - Py) - y
        L.append(f"{nid}, 1, 1, {ux:.6e}")
        L.append(f"{nid}, 2, 2, {uy:.6e}")
        L.append(f"{nid}, 3, 3, 0.0")
    L.append("*EL PRINT, ELSET=EALL, TOTALS=ONLY")
    L.append("ELSE")
    L.append("*NODE FILE")
    L.append("U")
    L.append("*EL FILE")
    L.append("E")
    L.append("*END STEP")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def deploy_ccx(p, pivot, angle_deg, material=STEEL, imp_amp=None, elastic_only=False,
               n_through=1, lc_min=None, lc_max=None, n_frames=10, workdir="/tmp/ccx_job",
               hyp=None):
    """Single-step deployment (straight-line ramp). Kept for quick smoke tests."""
    mat, hyp = coerce_material(material), hyp or Hypotheses()
    lc_min = lc_min if lc_min is not None else max(p.w_c / 2, p.w_lig / 8)
    lc_max = lc_max if lc_max is not None else p.r_win / 4
    imp_amp = imp_amp if imp_amp is not None else 0.3 * p.thickness
    os.makedirs(workdir, exist_ok=True)
    job = os.path.join(workdir, "hinge")
    xyz, conn, arcA, arcB = _build_mesh(p, pivot, imp_amp, n_through, lc_min, lc_max)
    _write_inp(job + ".inp", xyz, conn, arcA, arcB, pivot, mat,
               np.radians(angle_deg), elastic_only, n_frames, hyp)
    r = subprocess.run([CCX_BIN, "hinge"], cwd=workdir, capture_output=True, text=True, timeout=600)
    return dict(returncode=r.returncode, stdout=r.stdout[-2000:], n_nodes=len(xyz),
                n_elems=len(conn), job=job)


# ── multi-step arc-path deck (correct large-rotation path) ──────────────────────

def _arc_disp(xyz, arcB, pivot, a, s, theta):
    """Prescribed displacement of each arc-B node for rigid motion (a, s, theta) about pivot."""
    c, si = np.cos(theta), np.sin(theta)
    Px, Py = pivot
    x = xyz[np.asarray(arcB) - 1, 0]
    y = xyz[np.asarray(arcB) - 1, 1]
    ux = Px + c * (x - Px) - si * (y - Py) - x + a
    uy = Py + si * (x - Px) + c * (y - Py) - y + s
    return ux, uy


def _states_at_times(states, times):
    """Imposed (a, s, theta) at each CalculiX total time -> (n_times, 3).

    One ``*STEP`` per state, so state ``k`` is reached at total time ``k+1`` and CalculiX ramps the
    prescribed boundary values linearly from the previous converged state across the step. Reading
    the label back is therefore a linear interpolation of the state list on the integer time grid,
    with the undeformed origin at t = 0 -- which is what makes an arbitrary polyline replayable:
    the increment labels follow the states actually imposed instead of assuming a straight ramp.
    """
    S = np.vstack([np.zeros(3), np.asarray(states, float)])      # row i is the state at time i
    grid = np.arange(len(S), dtype=float)
    t = np.clip(np.asarray(times, float), 0.0, grid[-1])
    return np.stack([np.interp(t, grid, S[:, j]) for j in range(3)], axis=1)


def _write_deck(path, xyz, conn, arcA, arcB, pivot, mat, states, elastic_only, field_every,
                solver=None, hyp=None, min_inc=1e-3, stabilize=None):
    """One *STEP per kinematic state (a,s,theta) -> correct arc path; energy+reaction+fields out.

    ``stabilize`` (None = off): add ``*STATIC, STABILIZE=<val>`` automatic viscous damping to walk
    through buckling snap-through (e.g. the fold past the localization wall). NOTE the viscous
    reaction contaminates F/M -- use only when the out-of-plane displacement (uz) is the target.
    """
    hyp = hyp or Hypotheses()
    L = ["*NODE"]
    for i, (x, y, z) in enumerate(xyz, start=1):
        L.append(f"{i}, {x:.6e}, {y:.6e}, {z:.6e}")
    L.append("*ELEMENT, TYPE=C3D15, ELSET=EALL")
    for e, row in enumerate(conn, start=1):
        L.append(f"{e}, " + ", ".join(str(v) for v in row))
    L.append("*NSET, NSET=ARCA\n" + ",\n".join(str(v) for v in arcA))
    L.append("*NSET, NSET=ARCB\n" + ",\n".join(str(v) for v in arcB))
    L.append(mat.constitutive_cards(hyp, elastic_only=elastic_only))
    L.append(mat.section_cards("EALL", hyp))
    stat = "*STATIC" + (f", SOLVER={solver}" if solver else "") \
         + (f", STABILIZE={stabilize:g}" if stabilize else "")
    # min increment (default 1e-3): the solver bails (ends the job) once it needs tiny steps -- for
    # steel that is the deep-plastic grind past rupture (natural stop-at-fracture, no endless
    # cutbacks). Thin elastic sheets (paper) buckle-snap instead and need a smaller floor to walk
    # through the fold -> tunable via ``min_inc``.
    for k, (a, s, th) in enumerate(states):
        L.append(f"*STEP, NLGEOM, INC=500\n{stat}\n0.25, 1.0, {min_inc:g}, 1.0\n*BOUNDARY")
        if k == 0:
            L.append("ARCA, 1, 3, 0.0")
        ux, uy = _arc_disp(xyz, arcB, pivot, a, s, th)
        for j, nid in enumerate(arcB):
            L.append(f"{nid}, 1, 1, {ux[j]:.6e}\n{nid}, 2, 2, {uy[j]:.6e}\n{nid}, 3, 3, 0.0")
        L.append("*EL PRINT, ELSET=EALL, TOTALS=ONLY\nELSE")
        L.append("*NODE PRINT, NSET=ARCB\nRF")
        if (k % field_every == 0) or (k == len(states) - 1):
            L.append("*NODE FILE\nU\n*EL FILE\n" + mat.el_file_fields(elastic_only=elastic_only))
        L.append("*END STEP")
    open(path, "w").write("\n".join(L) + "\n")


def _parse_dat(path):
    """Per output time: stored energy W and the raw arc-B reactions [(node, fx, fy), ...]."""
    res = {}
    mode, t = None, None
    for ln in open(path):
        if "total internal energy for set EALL and time" in ln:
            t = float(ln.split("time")[1]); mode = "E"; continue
        if "forces (fx,fy,fz) for set ARCB and time" in ln:
            t = float(ln.split("time")[1]); mode = "F"
            res.setdefault(t, {})["rf"] = []; continue
        s = ln.split()
        if mode == "E" and len(s) == 1:
            try:
                res.setdefault(t, {})["W"] = float(s[0]); mode = None
            except ValueError:
                pass
        elif mode == "F" and len(s) >= 4:
            try:
                res[t]["rf"].append((int(s[0]), float(s[1]), float(s[2])))
            except ValueError:
                mode = None
    return res


def _generalized_forces(rf, xyz, pivot, theta):
    """(F_a, F_s, M_theta) = reactions work-conjugate to (a,s,theta), using CURRENT coords."""
    Px, Py = pivot
    c, si = np.cos(theta), np.sin(theta)
    Fa = Fs = M = 0.0
    for nid, fx, fy in rf:
        X, Y = xyz[nid - 1, 0], xyz[nid - 1, 1]
        px = c * (X - Px) - si * (Y - Py)            # rotated lever arm rel. pivot (current config)
        py = si * (X - Px) + c * (Y - Py)
        Fa += fx; Fs += fy; M += px * fy - py * fx
    return Fa, Fs, M


_FRD_FLOAT = re.compile(r"[-+]?\d\.\d+E[-+]\d+")


def _parse_frd(path):
    """Per output frame: nodal DISP (N,3) and total strain TOSTRAIN (N,6). Ignores mesh block."""
    frames, cur, field, ncomp, data = [], None, None, 0, []
    for ln in open(path):
        tag = ln[:3]
        if tag == " -4":
            parts = ln.split()
            if len(parts) < 3:                        # truncated header (ccx killed mid-write) -> keep complete frames
                break
            field = parts[1]; ncomp = int(parts[2]); data = []
        elif tag == " -5":
            continue
        elif tag == " -1" and field:
            data.append([float(x) for x in _FRD_FLOAT.findall(ln[13:])])
        elif tag == " -2" and field and data:
            data[-1].extend(float(x) for x in _FRD_FLOAT.findall(ln[3:]))
        elif tag == " -3" and field:
            arr = (np.array([(v + [0.0] * ncomp)[:ncomp] for v in data])
                   if data else np.zeros((0, ncomp)))     # pad/truncate rows to ncomp
            if field == "DISP":
                if cur:
                    frames.append(cur)
                cur = {"DISP": arr}
            elif cur is not None:
                cur[field] = arr
            field = None
    if cur:
        frames.append(cur)
    return frames


def _principal_strain_max(tostrain):
    """Back-compat shim: per-element max principal strain (see damage.max_principal_strain)."""
    return max_principal_strain(tostrain)


def prepare_job(p, angle_deg=60.0, n_steps=15, pivot=None, material=STEEL, imp_amp=None,
                elastic_only=False, n_through=1, lc_min=None, lc_max=None, field_every=1,
                a=0.0, s=0.0, solver=None, workdir="/tmp/ccx_job", hyp=None, min_inc=1e-3,
                stabilize=None, states=None):
    """Build the mesh + write the deck (the gmsh part — NOT thread-safe, run serially).

    ``states``: an explicit list of ``(a, s, theta_rad)`` kinematic states to drive through, one
    ``*STEP`` each. Given, it REPLACES the proportional ramp synthesised from ``(a, s, angle_deg)``
    — this is how a measured hinge polyline is replayed verbatim. The oracle is elastoplastic, so
    ``W`` is path-dependent and the route matters, not only the destination.
    """
    mat, hyp = coerce_material(material), hyp or Hypotheses()
    lc_min = lc_min if lc_min is not None else max(p.w_c / 2, p.w_lig / 8)
    lc_max = lc_max if lc_max is not None else p.r_win / 4
    imp_amp = imp_amp if imp_amp is not None else 0.3 * p.thickness
    # pivot = the primary-cut tip = the energy-minimising rotation centre (user-locked hypothesis)
    pivot = pivot if pivot is not None else (0.0, -p.w_lig)
    os.makedirs(workdir, exist_ok=True)
    job = os.path.join(workdir, "hinge")
    xyz, conn, arcA, arcB = _build_mesh(p, pivot, imp_amp, n_through, lc_min, lc_max)
    if states is None:
        dth = np.radians(angle_deg) / n_steps
        states = [(a * (k + 1) / n_steps, s * (k + 1) / n_steps, (k + 1) * dth)
                  for k in range(n_steps)]
    states = np.asarray(states, float).reshape(-1, 3)
    _write_deck(job + ".inp", xyz, conn, arcA, arcB, pivot, mat, states, elastic_only,
                field_every, solver=solver, hyp=hyp, min_inc=min_inc, stabilize=stabilize)
    return dict(job=job, workdir=workdir, xyz=xyz, conn=conn, arcA=arcA, arcB=arcB, pivot=pivot,
                angle_deg=angle_deg, n_steps=len(states), states=states,
                material=mat, hyp=hyp, w_lig=p.w_lig)


def solve_job(meta, ncpus=1, timeout=1800, eps_f=None, fracture_margin=1.1, poll=4.0,
              ccx_bin=CCX_BIN):
    """Run ccx on a prepared job (subprocess — safe to run many concurrently).

    stop-at-fracture: if ``eps_f`` is given, poll the incrementally-written .frd and kill ccx once
    the hottest ligament element passes ``fracture_margin*eps_f``. This skips the deep-plastic
    grind PAST ductile failure — the dominant cost, since most hinges tear at small angle then ccx
    crawls to the cap in tiny increments (that data is also unphysical: CalculiX has no element
    deletion). Pass the MATERIAL's own ``eps_f``, not a campaign constant.
    Returns a CompletedProcess-like object (``.stdout`` = the ccx log).
    """
    env = {**os.environ, "OMP_NUM_THREADS": str(ncpus), "CCX_NPROC_EQUATION_SOLVER": str(ncpus)}
    if eps_f is None:
        return subprocess.run([ccx_bin, "hinge"], cwd=meta["workdir"], capture_output=True,
                              text=True, timeout=timeout, env=env)

    xyz, conn, w_lig = meta["xyz"], meta["conn"], meta["w_lig"]
    frd, logpath = meta["job"] + ".frd", os.path.join(meta["workdir"], "ccx.log")
    with open(logpath, "w") as logf:                             # file, not PIPE: no buffer deadlock
        proc = subprocess.Popen([ccx_bin, "hinge"], cwd=meta["workdir"], stdout=logf,
                                stderr=subprocess.STDOUT, text=True, env=env)
        t0 = time.time()
        while True:
            try:
                proc.wait(timeout=poll)
                break                                            # ccx finished on its own
            except subprocess.TimeoutExpired:
                pass
            if time.time() - t0 > timeout:
                proc.terminate(); break
            try:                                                 # peek the partial .frd for fracture
                frames = _parse_frd(frd) if os.path.exists(frd) else []
                if frames and peak_peeq(frames[-1], xyz, conn, w_lig) >= fracture_margin * eps_f:
                    proc.terminate(); break                      # stop-at-fracture
            except Exception:
                pass
        if proc.poll() is None:
            try:
                proc.wait(timeout=20)
            except Exception:
                proc.kill()
    with open(logpath) as f:
        out = f.read()
    return subprocess.CompletedProcess([ccx_bin, "hinge"], proc.returncode or 0, out, "")


def parse_job(meta, stdout=""):
    """Read energy / forces / fields from a solved job."""
    job, xyz, pivot = meta["job"], meta["xyz"], meta["pivot"]
    ok = "Job finished" in stdout
    dat = _parse_dat(job + ".dat")
    times = sorted(dat)
    # Label each increment from the states actually imposed. For a proportional ray this is
    # algebraically identical to the old t*angle/n_steps; for a replayed polyline it is the only
    # correct labelling, since (a, s) no longer track theta.
    u = _states_at_times(meta["states"], times)
    a_imp, s_imp, theta = u[:, 0], u[:, 1], u[:, 2]
    theta_deg = np.degrees(theta)
    W = np.array([dat[t].get("W", np.nan) for t in times])
    Fa, Fs, Mt = [], [], []
    for i, t in enumerate(times):
        rf = dat[t].get("rf", [])
        fa, fs, m = _generalized_forces(rf, xyz, pivot, theta[i]) \
            if rf else (np.nan, np.nan, np.nan)
        Fa.append(fa); Fs.append(fs); Mt.append(m)
    frames = _parse_frd(job + ".frd") if os.path.exists(job + ".frd") else []
    uz_max = np.array([np.abs(f["DISP"][:, 2]).max() for f in frames])
    strain_max = np.array([_principal_strain_max(f["TOSTRAIN"]).max()
                           for f in frames if "TOSTRAIN" in f])
    peeq_max = np.array([_peeq_max(f) for f in frames])
    # THE damage column: normalized plastic dissipation <PEEQ>_lig / eps_f (nff.rve.damage).
    # peeq_lig is the hottest ligament element -- it calibrates the tear line (the value of
    # `damage` where it first reaches eps_f); eta_mean_lig audits the constant-eps_f assumption.
    conn, w_lig = meta["conn"], meta["w_lig"]
    mat = coerce_material(meta["material"])
    hyp = meta.get("hyp") or Hypotheses()
    damage = np.array([mat.damage(f, hyp, xyz=xyz, conn=conn, w_lig=w_lig) for f in frames])
    peeq_lig = np.array([peak_peeq(f, xyz, conn, w_lig) for f in frames])
    eta_mean_lig = np.array([mean_triaxiality(f, xyz, conn, w_lig) for f in frames])
    return dict(ok=ok, stdout=stdout[-1500:], theta_deg=theta_deg, a=a_imp, s=s_imp,
                W=W, M_theta=np.array(Mt),
                F_a=np.array(Fa), F_s=np.array(Fs), uz_max=uz_max, strain_max=strain_max,
                peeq_max=peeq_max, peeq_lig=peeq_lig, damage=damage, eta_mean_lig=eta_mean_lig,
                eps_f=mat.eps_f, frames=frames, xyz=xyz, conn=conn,
                arcA=meta["arcA"], arcB=meta["arcB"], pivot=pivot, n_nodes=len(xyz),
                n_elems=len(conn), job=job)


def deploy(p, ncpus=1, solver=None, timeout=1800, eps_f=None, **kw):
    """Deploy one hinge (prepare + solve + parse). See prepare_job for kwargs.

    ``eps_f`` (ductile fracture strain) enables stop-at-fracture in solve_job. A timeout is
    tolerated: ccx writes the .frd/.dat incrementally, so we parse whatever increments
    completed before the kill (partial data up to fracture / where it stalled).
    """
    meta = prepare_job(p, solver=solver, **kw)
    try:
        stdout = solve_job(meta, ncpus=ncpus, timeout=timeout, eps_f=eps_f).stdout
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    return parse_job(meta, stdout)


def _peeq_max(frame):
    """Max equivalent plastic strain (PEEQ) anywhere in the frame; NaN if absent.

    Raw nodal max over the WHOLE mesh, so it is contaminated by the fillet stress singularity and
    by the coarse panels -- diagnostic only. The tear indicator is ``peeq_lig``
    (:func:`nff.rve.damage.peak_peeq`), which is an element mean inside the ligament.
    """
    for k in ("PE", "PEEQ"):
        if k in frame and frame[k].size:
            return float(np.abs(frame[k]).max())
    return np.nan

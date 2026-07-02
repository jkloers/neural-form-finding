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

import numpy as np

from nff.rve.geometry import RVEParams, build_rve_domain, boundary_tag

STEEL = dict(E=210_000.0, nu=0.30, sigma_y=235.0, Et=2100.0)      # S235, hardening E/100


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
        gmsh.model.mesh.field.setNumber(1, "YCenter", -0.5 * p.w_lig)   # mid-ligament
        gmsh.model.mesh.field.setNumber(1, "ZCenter", 0.5 * p.thickness)
        gmsh.model.mesh.field.setNumber(1, "Radius", 0.75 * p.w_lig)    # covers fillet + strip + secondary
        gmsh.model.mesh.field.setNumber(1, "Thickness", 0.75 * p.w_lig)  # transition to coarse
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


def _write_inp(path, xyz, conn, arcA, arcB, pivot, mat, angle, elastic_only, n_frames):
    L = ["*NODE"]
    for i, (x, y, z) in enumerate(xyz, start=1):
        L.append(f"{i}, {x:.6e}, {y:.6e}, {z:.6e}")
    L.append("*ELEMENT, TYPE=C3D15, ELSET=EALL")
    for e, row in enumerate(conn, start=1):
        L.append(f"{e}, " + ", ".join(str(v) for v in row))
    L.append("*NSET, NSET=ARCA\n" + ",\n".join(str(v) for v in arcA))
    L.append("*MATERIAL, NAME=STEEL")
    L.append("*ELASTIC")
    L.append(f"{mat['E']:.1f}, {mat['nu']:.3f}")
    if not elastic_only:
        L.append("*PLASTIC")
        L.append(f"{mat['sigma_y']:.1f}, 0.0")
        L.append(f"{mat['sigma_y'] + mat['Et'] * 0.5:.1f}, 0.5")  # hardening slope Et
    L.append("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL")
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
               n_through=1, lc_min=None, lc_max=None, n_frames=10, workdir="/tmp/ccx_job"):
    """Single-step deployment (straight-line ramp). Kept for quick smoke tests."""
    lc_min = lc_min if lc_min is not None else max(p.w_c / 2, p.w_lig / 8)
    lc_max = lc_max if lc_max is not None else p.r_win / 4
    imp_amp = imp_amp if imp_amp is not None else 0.3 * p.thickness
    os.makedirs(workdir, exist_ok=True)
    job = os.path.join(workdir, "hinge")
    xyz, conn, arcA, arcB = _build_mesh(p, pivot, imp_amp, n_through, lc_min, lc_max)
    _write_inp(job + ".inp", xyz, conn, arcA, arcB, pivot, material,
               np.radians(angle_deg), elastic_only, n_frames)
    r = subprocess.run(["ccx", "hinge"], cwd=workdir, capture_output=True, text=True, timeout=600)
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


def _write_deck(path, xyz, conn, arcA, arcB, pivot, mat, states, elastic_only, field_every,
                solver=None):
    """One *STEP per kinematic state (a,s,theta) -> correct arc path; energy+reaction+fields out."""
    L = ["*NODE"]
    for i, (x, y, z) in enumerate(xyz, start=1):
        L.append(f"{i}, {x:.6e}, {y:.6e}, {z:.6e}")
    L.append("*ELEMENT, TYPE=C3D15, ELSET=EALL")
    for e, row in enumerate(conn, start=1):
        L.append(f"{e}, " + ", ".join(str(v) for v in row))
    L.append("*NSET, NSET=ARCA\n" + ",\n".join(str(v) for v in arcA))
    L.append("*NSET, NSET=ARCB\n" + ",\n".join(str(v) for v in arcB))
    L.append("*MATERIAL, NAME=STEEL\n*ELASTIC")
    L.append(f"{mat['E']:.1f}, {mat['nu']:.3f}")
    if not elastic_only:
        L.append("*PLASTIC")
        L.append(f"{mat['sigma_y']:.1f}, 0.0")
        L.append(f"{mat['sigma_y'] + mat['Et'] * 0.5:.1f}, 0.5")
    L.append("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL")
    stat = "*STATIC" + (f", SOLVER={solver}" if solver else "")
    # min increment 1e-3: the solver bails (ends the job) once it needs tiny steps -- which is
    # exactly the deep-plastic grind past rupture -> natural stop-at-fracture, no endless cutbacks.
    for k, (a, s, th) in enumerate(states):
        L.append(f"*STEP, NLGEOM, INC=200\n{stat}\n0.25, 1.0, 1e-3, 1.0\n*BOUNDARY")
        if k == 0:
            L.append("ARCA, 1, 3, 0.0")
        ux, uy = _arc_disp(xyz, arcB, pivot, a, s, th)
        for j, nid in enumerate(arcB):
            L.append(f"{nid}, 1, 1, {ux[j]:.6e}\n{nid}, 2, 2, {uy[j]:.6e}\n{nid}, 3, 3, 0.0")
        L.append("*EL PRINT, ELSET=EALL, TOTALS=ONLY\nELSE")
        L.append("*NODE PRINT, NSET=ARCB\nRF")
        if (k % field_every == 0) or (k == len(states) - 1):
            L.append("*NODE FILE\nU\n*EL FILE\n" + ("E, S" if elastic_only else "E, PEEQ, S"))
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
            field = ln.split()[1]; ncomp = int(ln.split()[2]); data = []
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
    """Max principal strain per frame from TOSTRAIN (exx,eyy,ezz,exy,eyz,ezx)."""
    e = tostrain
    out = np.zeros(len(e))
    for i, (xx, yy, zz, xy, yz, zx) in enumerate(e):
        T = np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])
        out[i] = np.linalg.eigvalsh(T)[-1]
    return out


def prepare_job(p, angle_deg=60.0, n_steps=15, pivot=None, material=STEEL, imp_amp=None,
                elastic_only=False, n_through=1, lc_min=None, lc_max=None, field_every=1,
                a=0.0, s=0.0, solver=None, workdir="/tmp/ccx_job"):
    """Build the mesh + write the deck (the gmsh part — NOT thread-safe, run serially)."""
    lc_min = lc_min if lc_min is not None else max(p.w_c / 2, p.w_lig / 8)
    lc_max = lc_max if lc_max is not None else p.r_win / 4
    imp_amp = imp_amp if imp_amp is not None else 0.3 * p.thickness
    # pivot = the primary-cut tip = the energy-minimising rotation centre (user-locked hypothesis)
    pivot = pivot if pivot is not None else (0.0, -p.w_lig)
    os.makedirs(workdir, exist_ok=True)
    job = os.path.join(workdir, "hinge")
    xyz, conn, arcA, arcB = _build_mesh(p, pivot, imp_amp, n_through, lc_min, lc_max)
    dth = np.radians(angle_deg) / n_steps
    states = [(a * (k + 1) / n_steps, s * (k + 1) / n_steps, (k + 1) * dth) for k in range(n_steps)]
    _write_deck(job + ".inp", xyz, conn, arcA, arcB, pivot, material, states, elastic_only,
                field_every, solver=solver)
    return dict(job=job, workdir=workdir, xyz=xyz, conn=conn, arcA=arcA, arcB=arcB, pivot=pivot,
                angle_deg=angle_deg, n_steps=n_steps)


def solve_job(meta, ncpus=1, timeout=1800):
    """Run ccx on a prepared job (subprocess — safe to run many concurrently)."""
    env = {**os.environ, "OMP_NUM_THREADS": str(ncpus), "CCX_NPROC_EQUATION_SOLVER": str(ncpus)}
    return subprocess.run(["ccx", "hinge"], cwd=meta["workdir"], capture_output=True, text=True,
                          timeout=timeout, env=env)


def parse_job(meta, stdout=""):
    """Read energy / forces / fields from a solved job."""
    job, xyz, pivot = meta["job"], meta["xyz"], meta["pivot"]
    angle_deg, n_steps = meta["angle_deg"], meta["n_steps"]
    ok = "Job finished" in stdout
    dat = _parse_dat(job + ".dat")
    times = sorted(dat)
    theta_deg = np.array([t * angle_deg / n_steps for t in times])
    W = np.array([dat[t].get("W", np.nan) for t in times])
    Fa, Fs, Mt = [], [], []
    for t in times:
        rf = dat[t].get("rf", [])
        fa, fs, m = _generalized_forces(rf, xyz, pivot, np.radians(t * angle_deg / n_steps)) \
            if rf else (np.nan, np.nan, np.nan)
        Fa.append(fa); Fs.append(fs); Mt.append(m)
    frames = _parse_frd(job + ".frd") if os.path.exists(job + ".frd") else []
    uz_max = np.array([np.abs(f["DISP"][:, 2]).max() for f in frames])
    strain_max = np.array([_principal_strain_max(f["TOSTRAIN"]).max()
                           for f in frames if "TOSTRAIN" in f])
    peeq_max = np.array([_peeq_max(f) for f in frames])
    return dict(ok=ok, stdout=stdout[-1500:], theta_deg=theta_deg, W=W, M_theta=np.array(Mt),
                F_a=np.array(Fa), F_s=np.array(Fs), uz_max=uz_max, strain_max=strain_max,
                peeq_max=peeq_max, frames=frames, xyz=xyz, conn=meta["conn"], arcA=meta["arcA"],
                arcB=meta["arcB"], pivot=pivot, n_nodes=len(xyz), n_elems=len(meta["conn"]), job=job)


def deploy(p, ncpus=1, solver=None, timeout=1800, **kw):
    """Deploy one hinge (prepare + solve + parse). See prepare_job for kwargs.

    A timeout is tolerated: ccx writes the .frd/.dat incrementally, so we parse whatever
    increments completed before the kill (partial data up to where it stalled).
    """
    meta = prepare_job(p, solver=solver, **kw)
    try:
        stdout = solve_job(meta, ncpus=ncpus, timeout=timeout).stdout
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    return parse_job(meta, stdout)


def _peeq_max(frame):
    """Robust max equivalent plastic strain (PEEQ) for the failure flag; NaN if absent."""
    for k in ("PE", "PEEQ"):
        if k in frame and frame[k].size:
            return float(np.abs(frame[k]).max())
    return np.nan

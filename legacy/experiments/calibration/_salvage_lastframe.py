"""Stream the LAST field frame out of a huge .frd and report where the plastic work sits.

A campaign .frd is ~1.2 GB (fields at every increment), so the ordinary parser -- which builds
every frame in memory -- cannot be used casually. This keeps only the frame being read.
"""
from __future__ import annotations

import re
import sys

import numpy as np

from nff.rve.damage import _element_mean, element_volumes, ligament_elements

_FRD_FLOAT = re.compile(r"[-+]?\d\.\d+E[-+]\d+")


def last_frame(path):
    """The final complete output frame, as {field: (N, ncomp)}, streaming one frame at a time."""
    cur, prev, field, ncomp, data = None, None, None, 0, []
    for ln in open(path):
        tag = ln[:3]
        if tag == " -4":
            parts = ln.split()
            if len(parts) < 3:
                break
            field, ncomp, data = parts[1], int(parts[2]), []
        elif tag == " -1" and field:
            data.append([float(x) for x in _FRD_FLOAT.findall(ln[13:])])
        elif tag == " -2" and field and data:
            data[-1].extend(float(x) for x in _FRD_FLOAT.findall(ln[3:]))
        elif tag == " -3" and field:
            arr = (np.array([(v + [0.0] * ncomp)[:ncomp] for v in data])
                   if data else np.zeros((0, ncomp)))
            if field == "DISP":
                if cur:
                    prev = cur
                cur = {"DISP": arr}
            elif cur is not None:
                cur[field] = arr
            field = None
    return cur if cur and len(cur) > 1 else prev


def read_mesh(inp_path):
    """Nodes + C3D15 connectivity straight out of the deck we wrote."""
    xyz, conn, mode = [], [], None
    for ln in open(inp_path):
        if ln.startswith("*"):
            u = ln.upper()
            mode = "n" if u.startswith("*NODE") and "FILE" not in u and "PRINT" not in u else (
                "e" if u.startswith("*ELEMENT") else None)
            continue
        if mode == "n":
            p = ln.split(",")
            if len(p) >= 4:
                xyz.append([float(p[1]), float(p[2]), float(p[3])])
        elif mode == "e":
            p = [q.strip() for q in ln.split(",") if q.strip()]
            if len(p) == 16:
                conn.append([int(q) for q in p[1:]])
    return np.array(xyz), np.array(conn, int)


def main():
    w_lig = float(sys.argv[1])
    for job in sys.argv[2:]:
        xyz, conn = read_mesh(job + "/hinge.inp")
        f = last_frame(job + "/hinge.frd")
        if f is None:
            print(f"{job}: no complete frame"); continue
        nodal = None
        for k in ("PEEQ", "PE"):
            if k in f and np.size(f[k]):
                a = np.abs(np.asarray(f[k], float))
                nodal = a[:, 0] if a.ndim > 1 else a
                break
        if nodal is None or len(nodal) != len(xyz):
            print(f"{job}: no usable PEEQ ({None if nodal is None else len(nodal)} vs {len(xyz)})")
            continue
        pe = _element_mean(nodal, conn)
        vol = element_volumes(xyz, conn)
        lig = ligament_elements(xyz, conn, w_lig)
        c = xyz[conn[:, :6] - 1].mean(axis=1)
        r = np.hypot(c[:, 0], c[:, 1] + 0.5 * w_lig)
        wk = vol * pe
        ring = (~lig) & (r > 90.0)
        uz = float(np.abs(f["DISP"][:, 2]).max())
        print(f"{job.split('/')[-1]:22s} lig_work {100*wk[lig].sum()/max(wk.sum(),1e-30):5.1f}%  "
              f"outer_ring {100*wk[ring].sum()/max(wk.sum(),1e-30):5.1f}%  "
              f"PEEQ lig_max {pe[lig].max():.4f}  far_max {pe[~lig].max():.4f}  "
              f"uz_max {uz:6.2f} mm  Delta {wk[lig].sum()/vol[lig].sum()/1.784:.5f}")


if __name__ == "__main__":
    main()

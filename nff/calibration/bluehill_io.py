"""Parse Instron Bluehill Universal tensile CSV exports.

The Bluehill "results + raw data" CSV is a multi-block text file::

    <specimen-properties header row>          # Test:Rate 1, Specimen properties:..., General:...
    <units row>
    <one properties value row>                # KNOWN to carry stale/sample-level metadata

    Results Table 1
    ,W_mean,T_mean,A_mean,t1,t2,t3,w1,w2,w3
    ,(mm),(mm),(mm^2),...
    "1", <per-specimen geometry ...>
    "2", ...

    1,Time,Displacement,Force                 # per-specimen raw curve block
    ,(s),(mm),(kN)
    ,"0.0000","0.0142","0.0031"
    ...
    2,Time,Displacement,Force
    ...

Only the per-specimen Results-Table geometry and the raw curve blocks are trusted.
The top properties row carries stale metadata in some exports (protocol §13), and the
per-specimen widths may not have been updated at test time — so the authoritative
per-specimen width / gauge length / direction come from the operator's manual summary
sheet (:mod:`summary_io`), matched to raw blocks BY ORDER.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np


@dataclass
class SpecimenRun:
    """One specimen's raw curve plus (untrusted) Bluehill geometry."""

    index: int
    time_s: np.ndarray          # (n,)
    disp_mm: np.ndarray         # (n,) crosshead displacement
    force_N: np.ndarray         # (n,)
    w_mean_mm: float | None = None      # from Results Table — prefer summary sheet
    t_mean_mm: float | None = None
    a_mean_mm2: float | None = None

    @property
    def n_points(self) -> int:
        return int(self.force_N.size)


def _to_float(s: str) -> float | None:
    """Tolerant float parse: strips quotes and accepts French decimal commas."""
    s = s.strip().strip('"')
    if s.count(",") == 1 and "." not in s:   # "0,5" -> "0.5"
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _is_specimen_index(cell: str) -> bool:
    return cell.strip().strip('"').isdigit()


def load_raw_csv(path: str, index: int = 1) -> SpecimenRun:
    """Parse a plain single-specimen ``Time,Displacement,Force`` export (one file/specimen).

    Newer Bluehill exports drop the block wrapper and write just a 3-column raw file with a
    header row and a units row. Geometry is not in the file — supply it from the summary sheet.
    """
    with open(path, encoding="latin-1", newline="") as f:
        rows = list(csv.reader(f))
    t, d, force = [], [], []
    for r in rows[2:]:  # skip header + units rows
        if len(r) < 3:
            continue
        tv, dv, fv = _to_float(r[0]), _to_float(r[1]), _to_float(r[2])
        if tv is None or dv is None or fv is None:
            continue
        t.append(tv)
        d.append(dv)
        force.append(fv * 1000.0)  # kN -> N
    return SpecimenRun(index, np.asarray(t), np.asarray(d), np.asarray(force))


def load_bluehill_csv(path: str) -> list[SpecimenRun]:
    """Parse a Bluehill results+raw CSV into per-specimen runs, ordered by index."""
    with open(path, encoding="latin-1", newline="") as f:
        rows = list(csv.reader(f))

    # --- per-specimen geometry from "Results Table 1" -----------------------
    geo: dict[int, list[float | None]] = {}
    for i, r in enumerate(rows):
        if r and r[0].strip() == "Results Table 1":
            j = i + 3  # skip the "Results Table 1" line, the column header, the units row
            while j < len(rows) and rows[j] and _is_specimen_index(rows[j][0]):
                idx = int(rows[j][0].strip().strip('"'))
                geo[idx] = [_to_float(x) for x in rows[j][1:4]]  # W_mean, T_mean, A_mean
                j += 1
            break

    # --- raw curve blocks: "<n>,Time,Displacement,Force" --------------------
    block_starts: list[tuple[int, int]] = []
    for i, r in enumerate(rows):
        if len(r) >= 4 and r[1].strip() == "Time" and r[2].strip() == "Displacement":
            block_starts.append((int(r[0].strip().strip('"')), i + 2))  # data starts after units row

    runs: list[SpecimenRun] = []
    for k, (idx, start) in enumerate(block_starts):
        end = block_starts[k + 1][1] - 2 if k + 1 < len(block_starts) else len(rows)
        t, d, force = [], [], []
        for r in rows[start:end]:
            if len(r) < 4:
                continue
            tv, dv, fv = _to_float(r[1]), _to_float(r[2]), _to_float(r[3])
            if tv is None or dv is None or fv is None:
                continue
            t.append(tv)
            d.append(dv)
            force.append(fv * 1000.0)  # kN -> N
        g = geo.get(idx, [None, None, None])
        runs.append(
            SpecimenRun(
                index=idx,
                time_s=np.asarray(t),
                disp_mm=np.asarray(d),
                force_N=np.asarray(force),
                w_mean_mm=g[0],
                t_mean_mm=g[1],
                a_mean_mm2=g[2],
            )
        )
    return runs

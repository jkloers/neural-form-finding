"""Parse the operator's manual summary sheet — the per-specimen source of truth.

Bluehill's per-specimen metadata is unreliable (stale header, un-updated widths,
empty labels — protocol §13), so the operator keeps a hand-maintained sheet with the
authoritative width / gauge length / direction / speed / name per specimen, matched to
the raw curve files by name (``Name`` -> ``<name with . -> _>.csv``).

Two layouts are supported (columns matched by fuzzy header name; French decimals OK)::

    L, t1..t3, w1..w3, T_Mean, W_Mean, A_Mean, Speed, Status, Name, Direction
    L, t1..t3, w1..w3, T_Mean, W_Mean, A_Mean, Speed, Name, Direction,
        Drawn width, Drawn thickness, Drawn Area, Lambda       # cold-draw batch

The sheet's own ``Lambda`` column is A_drawn/A0; the true draw ratio (length stretch)
is A0/A_drawn and is exposed as :attr:`SummaryRow.lam`.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass


@dataclass
class SummaryRow:
    name: str
    direction: str | None
    speed: str | None
    status: str | None
    L_mm: float | None
    t_mean_mm: float | None
    w_mean_mm: float | None
    a_mean_mm2: float | None
    drawn_w_mm: float | None = None
    drawn_t_mm: float | None = None
    drawn_a_mm2: float | None = None

    @property
    def included(self) -> bool:
        """True unless a Status column explicitly excludes the row."""
        if self.status is None:
            return True
        return not self.status.strip().lower().startswith(("not", "exclud"))

    @property
    def lam(self) -> float | None:
        """True natural draw ratio lambda = A0 / A_drawn (length stretch)."""
        if self.a_mean_mm2 and self.drawn_a_mm2:
            return self.a_mean_mm2 / self.drawn_a_mm2
        return None


def _f(s: str) -> float | None:
    s = (s or "").strip().strip('"').replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _find(header: list[str], *keys: str) -> int | None:
    for i, h in enumerate(header):
        hl = h.strip().lower()
        if any(k in hl for k in keys):
            return i
    return None


def load_summary(path: str) -> list[SummaryRow]:
    """Parse the summary sheet; returns all rows (filter with ``.included``)."""
    with open(path, encoding="latin-1", newline="") as f:
        rows = list(csv.reader(f))

    header_i = next(
        (i for i, r in enumerate(rows)
         if _find(r, "name") is not None and _find(r, "direction") is not None),
        None,
    )
    if header_i is None:
        raise ValueError(f"no header row with Name+Direction found in {path}")
    h = rows[header_i]
    col = {
        "L": _find(h, "l ") if _find(h, "l ") is not None else 0,
        "t": _find(h, "t_mean", "t mean"),
        "w": _find(h, "w_mean", "w mean"),
        "a": _find(h, "a_mean", "a mean"),
        "speed": _find(h, "speed"),
        "status": _find(h, "status"),
        "name": _find(h, "name"),
        "dir": _find(h, "direction"),
        "dw": _find(h, "drawn width"),
        "dt": _find(h, "drawn thick"),
        "da": _find(h, "drawn area"),
    }

    def cell(r, key):
        i = col[key]
        return r[i] if i is not None and i < len(r) else ""

    out: list[SummaryRow] = []
    for r in rows[header_i + 1 :]:
        if not any(c.strip() for c in r):
            continue
        name = cell(r, "name").strip()
        if not name:
            continue
        out.append(
            SummaryRow(
                name=name,
                direction=(cell(r, "dir").strip() or None),
                speed=(cell(r, "speed").strip() or None),
                status=(cell(r, "status").strip() or None) if col["status"] is not None else None,
                L_mm=_f(cell(r, "L")),
                t_mean_mm=_f(cell(r, "t")),
                w_mean_mm=_f(cell(r, "w")),
                a_mean_mm2=_f(cell(r, "a")),
                drawn_w_mm=_f(cell(r, "dw")),
                drawn_t_mm=_f(cell(r, "dt")),
                drawn_a_mm2=_f(cell(r, "da")),
            )
        )
    return out

"""
Corridor clear-width verifier.

PLACEHOLDER THRESHOLD: 0.90 m minimum clear width.
PLACEHOLDER — generic/placeholder value, not a real statutory jurisdiction.
Results must not be read as evidence of real-world building-code compliance.
See LIMITATIONS.md.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon

# PLACEHOLDER — generic threshold, not a real jurisdiction (see LIMITATIONS.md)
_MIN_WIDTH_M: float = 0.90
_CLAUSE_ID: str = "CORR-W-01"

# Exactly "corridor" only; aliases (hallway, hall, passage) deferred to Day 3
# when real RPLAN type labels are confirmed — see DECISIONS.md 2026-08-26.
_CORRIDOR_TYPES: frozenset[str] = frozenset({"corridor"})

_SAMPLES_PER_EDGE: int = 10   # chord samples per polygon edge
_MIN_CHORD_LEN: float = 1e-6  # discard floating-point degenerate chord artefacts (m)
_FAR: float = 1e4             # chord ray half-length — well beyond any floor plan (m)


def check(plan: dict) -> list[dict]:
    """
    Check every room typed as a corridor against the minimum clear-width threshold.

    Returns
    -------
    list of audit records, one per corridor room:
        clause_id      : str
        room_id        : str
        measured_value : float  (metres; absent on ERROR)
        threshold      : float  (metres; absent on ERROR)
        margin         : float  (measured − threshold; negative = fail; absent on ERROR)
        status         : "PASS" | "FAIL" | "ERROR"
        detail         : str    (present only on ERROR)
    """
    results = []
    for room in plan.get("rooms", []):
        if room.get("type") not in _CORRIDOR_TYPES:
            continue
        rid = room["id"]
        raw = room.get("polygon", [])
        if len(raw) < 3:
            results.append({
                "clause_id": _CLAUSE_ID,
                "room_id": rid,
                "status": "ERROR",
                "detail": f"polygon has {len(raw)} vertices (need ≥ 3)",
            })
            continue
        poly = Polygon([tuple(p) for p in raw])
        if not poly.is_valid or poly.area < 1e-9:
            results.append({
                "clause_id": _CLAUSE_ID,
                "room_id": rid,
                "status": "ERROR",
                "detail": "degenerate or invalid polygon",
            })
            continue
        width = _min_clear_width(poly)
        margin = round(width - _MIN_WIDTH_M, 6)
        results.append({
            "clause_id": _CLAUSE_ID,
            "room_id": rid,
            "measured_value": round(width, 6),
            "threshold": _MIN_WIDTH_M,
            "margin": margin,
            "status": "PASS" if margin >= 0 else "FAIL",
        })
    return results


def _min_clear_width(polygon: Polygon) -> float:
    """
    Minimum clear width of a corridor polygon.

    Algorithm: for each polygon edge, cast _SAMPLES_PER_EDGE chord lines
    perpendicular to that edge (using the edge's inward normal).  Record
    every chord segment that falls inside the polygon; track the global
    minimum length.

    Why per-edge normals, not a single global MBR axis: an L-shaped
    corridor's turned arm runs roughly parallel to the MBR major axis, so
    MBR-axis cuts measure that arm's length, not its width.  Per-edge normals
    sample across each arm independently — each arm's own edges produce the
    correct cross-section direction for that arm.

    Shapely always normalises Polygon exterior rings to CCW orientation, so
    the inward normal is always the left-hand perpendicular of each directed
    edge (rotate edge_dir by +90°), with no per-polygon orientation check.
    """
    coords = list(polygon.exterior.coords)[:-1]  # drop repeated closing vertex
    n = len(coords)
    min_width = float("inf")

    for i in range(n):
        p1 = np.array(coords[i], dtype=float)
        p2 = np.array(coords[(i + 1) % n], dtype=float)
        edge_vec = p2 - p1
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 1e-9:
            continue  # degenerate edge — skip

        edge_dir = edge_vec / edge_len
        # Inward normal for CCW exterior: left-hand perpendicular of edge_dir.
        normal = np.array([-edge_dir[1], edge_dir[0]])

        for t in np.linspace(0.1, 0.9, _SAMPLES_PER_EDGE):
            sample = p1 + t * edge_vec
            chord = LineString([
                (sample[0] - _FAR * normal[0], sample[1] - _FAR * normal[1]),
                (sample[0] + _FAR * normal[0], sample[1] + _FAR * normal[1]),
            ])
            clipped = polygon.intersection(chord)
            if clipped.is_empty:
                continue
            for geom in getattr(clipped, "geoms", [clipped]):
                if hasattr(geom, "length") and geom.length > _MIN_CHORD_LEN:
                    min_width = min(min_width, geom.length)

    return min_width if min_width < float("inf") else 0.0

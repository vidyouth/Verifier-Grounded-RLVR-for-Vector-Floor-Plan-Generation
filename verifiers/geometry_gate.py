"""
Geometry gate verifier.

Checks structural validity of a floor-plan JSON:
  - Room polygons are simple (non-self-intersecting) and geometrically valid.
  - Room polygons are mutually non-overlapping (shared edges are fine).
  - Every opening's endpoints lie on its declared host-wall polygon edges.

This is a structural gate, not a statutory clause — it has no threshold record
shape.  It returns a pass/fail flag and a list of specific issues.

No placeholder thresholds are used here; validity checks are purely geometric.

host_wall schema (see DECISIONS.md 2026-08-26):
  An array of 1–2 room IDs the wall separates.
  - Interior: ["room_a", "room_b"] — wall shared between two rooms.
  - Exterior: ["exterior", "room_x"] — wall on the building boundary.
  "exterior" is a reserved pseudo-ID resolved to the boundary polygon's edges.
"""

from __future__ import annotations

from shapely.geometry import LineString, Point, Polygon
from shapely.validation import explain_validity

_WALL_SNAP_TOL: float = 1e-6   # metres — endpoint-on-edge tolerance
_OVERLAP_AREA_TOL: float = 1e-9  # m² — ignore floating-point shared-edge artefacts


def check(plan: dict) -> dict:
    """
    Parameters
    ----------
    plan : dict
        Floor-plan JSON conforming to the project schema (see DECISIONS.md).

    Returns
    -------
    dict:
        "pass"   : bool
        "issues" : list of {"code": str, "detail": str}
    """
    issues: list[dict] = []
    room_polys: dict[str, Polygon] = {}

    # ── 1. Room polygon validity ──────────────────────────────────────────────
    for room in plan.get("rooms", []):
        rid = room["id"]
        raw = room.get("polygon", [])
        if len(raw) < 3:
            issues.append({
                "code": "ROOM_DEGENERATE",
                "detail": f"room {rid!r}: {len(raw)} vertices (need ≥ 3)",
            })
            continue
        poly = Polygon([tuple(p) for p in raw])
        if not poly.is_valid:
            issues.append({
                "code": "ROOM_INVALID",
                "detail": f"room {rid!r}: {explain_validity(poly)}",
            })
        if not poly.is_simple:
            issues.append({
                "code": "ROOM_SELF_INTERSECT",
                "detail": f"room {rid!r}: polygon edges cross",
            })
        room_polys[rid] = poly

    # ── 2. Mutual non-overlap ─────────────────────────────────────────────────
    ids = list(room_polys)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            overlap = room_polys[a].intersection(room_polys[b]).area
            if overlap > _OVERLAP_AREA_TOL:
                issues.append({
                    "code": "ROOM_OVERLAP",
                    "detail": (
                        f"rooms {a!r} and {b!r} share {overlap:.6f} m² of interior area"
                    ),
                })

    # ── 3. Opening placement ──────────────────────────────────────────────────
    boundary_poly = _build_boundary(plan)
    for opening in plan.get("openings", []):
        oid = opening["id"]
        endpoints = opening.get("endpoints", [])
        host_wall = opening.get("host_wall", [])

        if len(endpoints) != 2:
            issues.append({
                "code": "OPENING_BAD_ENDPOINTS",
                "detail": f"opening {oid!r}: expected 2 endpoints, got {len(endpoints)}",
            })
            continue

        edges = _host_wall_edges(host_wall, room_polys, boundary_poly)
        if not edges:
            issues.append({
                "code": "OPENING_UNRESOLVED_HOST",
                "detail": (
                    f"opening {oid!r}: host_wall {host_wall!r} resolves to no polygon edges"
                ),
            })
            continue

        for label, pt_raw in [("start", endpoints[0]), ("end", endpoints[1])]:
            if not _point_on_edges(tuple(pt_raw), edges, _WALL_SNAP_TOL):
                issues.append({
                    "code": "OPENING_OFF_WALL",
                    "detail": (
                        f"opening {oid!r}: {label} endpoint {tuple(pt_raw)} is not on "
                        f"host_wall {host_wall!r} (tolerance {_WALL_SNAP_TOL} m)"
                    ),
                })

    return {"pass": len(issues) == 0, "issues": issues}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_boundary(plan: dict) -> Polygon | None:
    raw = plan.get("boundary", {}).get("polygon", [])
    if len(raw) < 3:
        return None
    return Polygon([tuple(p) for p in raw])


def _host_wall_edges(
    host_wall: list[str],
    room_polys: dict[str, Polygon],
    boundary_poly: Polygon | None,
) -> list[tuple]:
    """
    Resolve host_wall IDs to polygon edge segments.

    "exterior" → building boundary polygon edges.
    "<room_id>" → that room's polygon edges.
    Unknown IDs are silently skipped (caller detects empty result).
    """
    edges: list[tuple] = []
    for hid in host_wall:
        poly = boundary_poly if hid == "exterior" else room_polys.get(hid)
        if poly is None:
            continue
        coords = list(poly.exterior.coords)
        for k in range(len(coords) - 1):
            edges.append((coords[k], coords[k + 1]))
    return edges


def _point_on_edges(
    pt: tuple,
    edges: list[tuple],
    tol: float,
) -> bool:
    p = Point(pt)
    return any(p.distance(LineString([a, b])) <= tol for a, b in edges)

"""
ResPlan → project-schema conversion (Day 3 phase B).

One ResPlan plan dict (list entry of ResPlan.pkl, geometry = live Shapely objects)
is converted to the project's locked floor-plan JSON schema consumed by the
verifier bank:

    {
      "metadata":  {...},
      "boundary":  {"polygon": [[x,y],...], "entrance": [x,y]},
      "rooms":     [{"id","type","polygon"}],
      "openings":  [{"id","type","host_wall","endpoints","clear_width","swing"}],
      "relations": {"derived": [{"type","rooms"}]},
      "audit":     []
    }

Design decisions (see docs/DECISIONS.md 2026-08-30):
  - Exclusion filters: drop any plan with a `stair` polygon (multi-floor); keep
    only plans with 5–8 rooms counting living+bedroom+bathroom+kitchen+storage
    (balconies converted but NOT counted).
  - Coordinate normalization: per-plan scale = sqrt(inner_polygon_area_canvas /
    plan["area_m2"]); every coordinate is divided by that scale and translated so
    the boundary bbox min sits at the origin. `net_area` is unusable, not used.
  - Openings carry no room refs in ResPlan; host_wall is reconstructed by
    buffering each door/window/front_door polygon by 0.6 * wall_depth and taking
    the (up to) two rooms with the largest buffered-overlap area.
  - Adjacent rooms sit ~0.16 m apart across the wall band, so there is no shared
    boundary segment to host an opening. For each interior opening we perform a
    LOCAL per-door midline gap-close: a thin patch is unioned into each host room
    out to the gap midline, creating a short real shared segment exactly across
    the opening span. Opening endpoints + midpoint then lie on both host rooms'
    polygon boundaries, satisfying geometry_gate and keeping egress's covers()
    filter connected — with no change to any verifier.
  - Corridor synthesis was attempted and abandoned (residual interior area is the
    inter-room wall-gap web, not corridor shapes). No room is ever typed
    "corridor"; CORR-W-01 is a vacuous pass on this dataset. See FINDINGS.md
    2026-08-30.

This module has no side effects; `convert_plan` is a pure function.
"""

from __future__ import annotations

import math
from typing import Any

from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import nearest_points, unary_union

# ── Room-type keys ────────────────────────────────────────────────────────────
# Interior rooms carried into the output.  `stair` is an exclusion trigger and
# never converted; garden/parking/pool are outdoor and dropped.
_ROOM_KEYS: tuple[str, ...] = ("living", "bedroom", "bathroom", "kitchen", "storage", "balcony")
# Rooms that count toward the 5–8 scope filter (balcony excluded by decision).
_COUNT_KEYS: tuple[str, ...] = ("living", "bedroom", "bathroom", "kitchen", "storage")

# ── Geometry constants (metres unless noted) ─────────────────────────────────
_HOST_EPS_WALL_FRAC: float = 0.6      # opening buffer = this * wall_depth_m
_GAP_CLOSE_MARGIN_M: float = 0.05     # patch over-reach past a room's facing edge
_MAX_GAP_CLOSE_M: float = 1.00        # skip the midline patch if the rooms are farther apart
                                     # (~4 wall thicknesses; wider openings left as-is)
_GAP_CLOSE_MAX_DEPTH_M: float = 1.20  # hard cap on patch depth (mis-resolved-door guard)
_MAX_ORIG_OVERLAP_M2: float = 0.5     # drop plan if any raw room pair overlaps > this
_MAX_RESIDUAL_OVERLAP_M2: float = 1e-6  # drop plan if overlap remains after the fix
_MIN_ROOM_AREA_M2: float = 0.5        # a converted room smaller than this ⇒ drop plan
_COLLINEAR_TOL: float = 1e-7          # cross-product tol for collinear-vertex pruning
_COORD_GRID: float = 1e-6            # output coordinate grid (metres); shared edges snap exact

_SCHEMA_VERSION: str = "0.1"
_JURISDICTION: str = "placeholder/generic — not a real jurisdiction"


# ── Public API ───────────────────────────────────────────────────────────────

def exclusion_reason(rp: dict[str, Any]) -> str | None:
    """Return a drop reason if the raw ResPlan plan fails an exclusion filter, else None."""
    if _polys(rp.get("stair")):
        return "has_stair"
    n = sum(len(_polys(rp.get(k))) for k in _COUNT_KEYS)
    if not (5 <= n <= 8):
        return "room_count_out_of_scope"
    return None


def convert_plan(rp: dict[str, Any]) -> tuple[dict | None, str | None]:
    """
    Convert one raw ResPlan plan dict to the project schema.

    Returns
    -------
    (plan_json, None)        on success
    (None, "<drop_reason>")  when the plan is filtered out or unconvertible

    Drop reasons: has_stair, room_count_out_of_scope, no_area, bad_inner,
    invalid_room_geometry, room_overlap_large, room_overlap_residual,
    room_too_small, no_entrance, conversion_error.
    """
    reason = exclusion_reason(rp)
    if reason:
        return None, reason

    area_m2 = rp.get("area")
    if not area_m2 or area_m2 <= 0:
        return None, "no_area"

    try:
        inner_ring = _largest_inner_ring(rp)
        if inner_ring is None:
            return None, "bad_inner"
        scale = math.sqrt(inner_ring.area / area_m2)          # canvas units per metre
        if not math.isfinite(scale) or scale <= 0:
            return None, "no_area"
        wall_depth_m = float(rp.get("wall_depth") or 4.0) / scale

        # Normalising transform: canvas → metres, then translate bbox min to origin.
        bx0, by0, _, _ = inner_ring.bounds
        ox, oy = bx0 / scale, by0 / scale

        def T(poly: Polygon) -> Polygon:
            return Polygon([((x / scale) - ox, (y / scale) - oy) for x, y in poly.exterior.coords])

        boundary_poly = _clean_polygon(T(inner_ring))
        if boundary_poly is None:
            return None, "bad_inner"

        # ── Rooms ────────────────────────────────────────────────────────────
        rooms: list[dict] = []          # {"id","type","poly"(Polygon)}
        for key in _ROOM_KEYS:
            for i, g in enumerate(_polys(rp.get(key))):
                poly = _clean_polygon(T(g))
                if poly is None:
                    return None, "invalid_room_geometry"
                rooms.append({"id": f"{key}_{i}", "type": key, "poly": poly})
        if not rooms:
            return None, "room_count_out_of_scope"

        # ── Overlap fix (smaller-area room yields the overlap region) ─────────
        if not _resolve_overlaps(rooms):
            return None, "room_overlap_large"

        # ── Openings, pass 1: resolve hosts + gap-close (mutates room polys) ──
        # Endpoints are NOT fixed here — they are projected onto the FINAL room
        # polygons in pass 2, after all room mutations are done, so a later
        # overlap mop-up or ring cleanup can never move an endpoint off its wall.
        room_by_id = {r["id"]: r for r in rooms}
        specs: list[dict] = []  # {id, otype, host_wall, conn}

        fd_list = _polys(rp.get("front_door"))
        for i, g in enumerate(fd_list):
            fd = T(g)
            host = _resolve_hosts(fd, rooms, wall_depth_m, want=1) or _nearest_room(fd, rooms)
            if not host:
                continue
            specs.append({"id": f"front_door_{i}", "otype": "door",
                          "host_wall": ["exterior", host[0]], "conn": fd})

        for kind in ("door", "window"):
            for i, g in enumerate(_polys(rp.get(kind))):
                conn = T(g)
                hosts = _resolve_hosts(conn, rooms, wall_depth_m, want=2)
                if len(hosts) == 2:
                    # An opening between two rooms is walkable regardless of how
                    # ResPlan traced it: genuine interior windows are rare and
                    # ResPlan mis-labels many interior doors / open passages as
                    # windows (upstream promotes interior via_window → via_door
                    # too — see resplan_utils.py). Emit every interior opening as
                    # a "door" so topology / egress see the real circulation.
                    ra, rb = room_by_id[hosts[0]], room_by_id[hosts[1]]
                    _gap_close(conn, ra, rb, rooms, boundary_poly)
                    specs.append({"id": f"{kind}_{i}", "otype": "door",
                                  "host_wall": [ra["id"], rb["id"]], "conn": conn})
                elif len(hosts) == 1:
                    specs.append({"id": f"{kind}_{i}", "otype": kind,
                                  "host_wall": ["exterior", hosts[0]], "conn": conn})
                # 0 hosts → opening dropped (the driver tallies this via the count diff)

        if not any("front_door" in s["id"] for s in specs):
            return None, "no_entrance"

        # ── Finalise room polygons: clean rings first, THEN mop up any sliver
        #     overlaps the gap-close introduced (order matters — cleaning after a
        #     subtraction can re-inflate an edge and reopen a sub-tolerance
        #     overlap that geometry_gate would then flag) ────────────────────────
        for r in rooms:
            cleaned = _clean_polygon(r["poly"])
            if cleaned is None or cleaned.area < _MIN_ROOM_AREA_M2:
                return None, "room_too_small"
            r["poly"] = _round_poly(cleaned)
            if r["poly"] is None:
                return None, "invalid_room_geometry"
        # Coordinate rounding itself can nudge a vertex onto a neighbour's edge and
        # open a sub-round-off overlap lens; scrub it, then re-round the scrub
        # output (the 1e-6 bite exceeds twice the 5e-7 rounding shift, so the cut
        # survives the second rounding).
        _scrub_overlap_dust(rooms)
        for r in rooms:
            r["poly"] = _round_poly(r["poly"])
            if r["poly"] is None:
                return None, "invalid_room_geometry"
        if not _no_residual_overlap(rooms, tol=1e-9):  # match geometry_gate's tolerance
            return None, "room_overlap_residual"
        room_by_id = {r["id"]: r for r in rooms}

        # ── Openings, pass 2: project endpoints onto the final polygons ──────
        openings: list[dict] = []
        derived: list[dict] = []
        entrance_xy: tuple[float, float] | None = None
        for s in specs:
            hw = s["host_wall"]
            if hw[0] == "exterior":
                target = boundary_poly.exterior if s["id"].startswith("front_door") \
                    else room_by_id[hw[1]]["poly"].exterior
            else:
                # Prefer the segment the two host rooms actually share (gap-close
                # created it): projecting the opening there puts its midpoint on
                # BOTH rooms' boundaries, so egress's covers() filter connects
                # from either side. Fall back to hw[0]'s edge if they don't meet.
                shared = room_by_id[hw[0]]["poly"].boundary.intersection(
                    room_by_id[hw[1]]["poly"].boundary
                )
                target = shared if getattr(shared, "length", 0.0) > 1e-6 \
                    else room_by_id[hw[0]]["poly"].exterior
            e1, e2, cw = _project_span(s["conn"], target)
            openings.append(_opening(s["id"], s["otype"], hw, e1, e2, cw))
            if s["id"].startswith("front_door"):
                entrance_xy = ((e1[0] + e2[0]) / 2.0, (e1[1] + e2[1]) / 2.0)
            if s["otype"] == "door":
                derived.append(
                    {"type": "direct" if hw[0] == "exterior" else "via_door", "rooms": list(hw)}
                )
        if entrance_xy is None:
            return None, "no_entrance"

        plan_json = {
            "metadata": {
                "schema_version": _SCHEMA_VERSION,
                "jurisdiction": _JURISDICTION,
                "source": {
                    "dataset": "ResPlan",
                    "plan_id": rp.get("id"),
                    "license": "CC BY 4.0",
                },
                "scale_canvas_units_per_m": round(scale, 8),
                "room_count_in_scope": sum(
                    1 for r in rooms if r["type"] in _COUNT_KEYS
                ),
            },
            "boundary": {
                "polygon": _ring_coords(boundary_poly),
                "entrance": [round(entrance_xy[0], 6), round(entrance_xy[1], 6)],
            },
            "rooms": [
                {"id": r["id"], "type": r["type"], "polygon": _ring_coords(r["poly"])}
                for r in rooms
            ],
            "openings": openings,
            "relations": {"derived": derived},
            "audit": [],
        }
        return plan_json, None

    except Exception:  # noqa: BLE001 — any geometry blow-up ⇒ drop the plan, don't crash the run
        return None, "conversion_error"


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _polys(v: Any) -> list[Polygon]:
    """Extract non-empty positive-area polygons from a ResPlan geometry value."""
    if v is None:
        return []
    if isinstance(v, Polygon):
        return [v] if (not v.is_empty and v.area > 0) else []
    if isinstance(v, MultiPolygon):
        return [g for g in v.geoms if not g.is_empty and g.area > 0]
    return []


def _largest_inner_ring(rp: dict[str, Any]) -> Polygon | None:
    """Largest exterior ring of the `inner` shell (holes dropped, multipart resolved)."""
    parts = _polys(rp.get("inner"))
    if not parts:
        return None
    biggest = max(parts, key=lambda g: g.area)
    return Polygon(biggest.exterior)


def _clean_polygon(poly: Polygon) -> Polygon | None:
    """Repair validity, take the largest part, drop holes, prune collinear vertices."""
    if poly is None or poly.is_empty:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if not isinstance(poly, Polygon):
        return None
    ring = list(poly.exterior.coords)[:-1]
    pruned: list[tuple[float, float]] = []
    n = len(ring)
    for i in range(n):
        a, b, c = ring[i - 1], ring[i], ring[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) < _COLLINEAR_TOL:
            continue
        if pruned and math.dist(pruned[-1], b) < 1e-9:
            continue
        pruned.append((b[0], b[1]))
    if len(pruned) < 3:
        return None
    out = Polygon(pruned)
    if not out.is_valid:
        out = out.buffer(0)
    return out if (isinstance(out, Polygon) and out.is_valid and out.area > 1e-9) else None


_COORD_DP: int = 6  # output coordinate decimal places (metres); 1e-6 m = 1 micron


def _round_poly(poly: Polygon) -> Polygon | None:
    """Round a polygon's coordinates to _COORD_DP and return a valid polygon
    (or None if the rounding degenerates it)."""
    if poly is None or poly.is_empty:
        return None
    ring = [(round(x, _COORD_DP), round(y, _COORD_DP))
            for x, y in list(poly.exterior.coords)[:-1]]
    if len(ring) < 3:
        return None
    out = Polygon(ring)
    if not out.is_valid:
        out = out.buffer(0)
    if isinstance(out, MultiPolygon) and out.geoms:
        out = max(out.geoms, key=lambda g: g.area)
    return out if (isinstance(out, Polygon) and out.is_valid and out.area > 1e-9) else None


def _ring_coords(poly: Polygon) -> list[list[float]]:
    return [[round(x, _COORD_DP), round(y, _COORD_DP)]
            for x, y in list(poly.exterior.coords)[:-1]]


def _resolve_overlaps(rooms: list[dict]) -> bool:
    """
    Subtract pairwise interior overlaps in-place; the smaller-area room yields.

    Returns False if any raw pair overlaps by more than _MAX_ORIG_OVERLAP_M2
    (caller drops the plan); True otherwise.
    """
    order = sorted(range(len(rooms)), key=lambda k: rooms[k]["poly"].area)  # small → large
    for pos, small_idx in enumerate(order):
        small = rooms[small_idx]["poly"]
        for big_idx in order[pos + 1:]:
            big = rooms[big_idx]["poly"]
            inter = small.intersection(big).area
            if inter <= 1e-9:
                continue
            if inter > _MAX_ORIG_OVERLAP_M2:
                return False
            small = small.difference(big)
            if isinstance(small, MultiPolygon):
                small = max(small.geoms, key=lambda g: g.area) if small.geoms else small
        if isinstance(small, Polygon) and not small.is_empty:
            rooms[small_idx]["poly"] = small
    return True


def _scrub_overlap_dust(rooms: list[dict]) -> None:
    """
    Final pass: subtract any remaining room-room overlap (down to floating-point
    dust) from the smaller-area room, so the output clears geometry_gate's 1e-9
    m² tolerance. The gap-close patch unions leave ~1e-8 m² slivers where two
    patches meet at the midline. The bite is `buffer(_COORD_GRID)` so the cut
    still clears after `_ring_coords` rounds coordinates to 6 dp (which can shift
    a vertex by up to 5e-7 m). Sub-micron shrink of the smaller room — immaterial.
    """
    for _ in range(3):
        order = sorted(range(len(rooms)), key=lambda k: rooms[k]["poly"].area)
        dirty = False
        for pos, si in enumerate(order):
            s = rooms[si]["poly"]
            for bi in order[pos + 1:]:
                if s.intersection(rooms[bi]["poly"]).area > 1e-12:
                    s = s.difference(rooms[bi]["poly"].buffer(_COORD_GRID))
                    dirty = True
            if isinstance(s, MultiPolygon):
                s = max(s.geoms, key=lambda g: g.area) if s.geoms else s
            if isinstance(s, Polygon) and not s.is_empty and s.is_valid:
                rooms[si]["poly"] = s
        if not dirty:
            return


def _no_residual_overlap(rooms: list[dict], tol: float = _MAX_RESIDUAL_OVERLAP_M2) -> bool:
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if rooms[i]["poly"].intersection(rooms[j]["poly"]).area > tol:
                return False
    return True


def _rect_dims(poly: Polygon) -> tuple[float, float, tuple[float, float]]:
    """(short_side, long_side, unit_vector_along_long_side) of the min rotated rect."""
    mrr = poly.minimum_rotated_rectangle
    if not hasattr(mrr, "exterior"):
        return 0.0, 0.0, (1.0, 0.0)
    xs = list(mrr.exterior.coords)
    sides = [(math.dist(xs[k], xs[k + 1]), k) for k in range(4)]
    long_len, lk = max(sides)
    short_len = min(s for s, _ in sides)
    dx, dy = xs[lk + 1][0] - xs[lk][0], xs[lk + 1][1] - xs[lk][1]
    d = math.hypot(dx, dy) or 1.0
    return short_len, long_len, (dx / d, dy / d)


def _resolve_hosts(conn: Polygon, rooms: list[dict], wall_depth_m: float, want: int) -> list[str]:
    """
    Buffer the connector by 0.6*wall_depth and return up to `want` room IDs,
    ranked by buffered-overlap area (deterministic: area desc, then id).
    """
    eps = max(_HOST_EPS_WALL_FRAC * wall_depth_m, 1e-4)
    b = conn.buffer(eps)
    scored: list[tuple[float, str]] = []
    for r in rooms:
        ov = b.intersection(r["poly"]).area
        if ov > 1e-9:
            scored.append((ov, r["id"]))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [rid for _, rid in scored[:want]]


def _nearest_room(conn: Polygon, rooms: list[dict]) -> list[str]:
    if not rooms:
        return []
    c = conn.centroid
    return [min(rooms, key=lambda r: r["poly"].distance(c))["id"]]


def _project_span(conn: Polygon, target) -> tuple[list[float], list[float], float]:
    """
    Opening endpoints = the connector's clear span, projected onto `target`
    (a LineString — a room-exterior ring or the shared segment between two host
    rooms). The returned points lie exactly on one of the opening's host_wall
    edges, satisfying geometry_gate; projecting onto the shared segment also puts
    the midpoint on both rooms so egress's covers() filter connects either side.
    """
    _, width, u = _rect_dims(conn)
    c = conn.centroid
    half = width / 2.0
    raw = [
        (c.x - half * u[0], c.y - half * u[1]),
        (c.x + half * u[0], c.y + half * u[1]),
    ]
    snapped = [nearest_points(target, Point(p))[0] for p in raw]
    return [round(snapped[0].x, 6), round(snapped[0].y, 6)], \
           [round(snapped[1].x, 6), round(snapped[1].y, 6)], round(width, 4)


def _gap_close(
    conn: Polygon,
    ra: dict,
    rb: dict,
    rooms: list[dict],
    boundary_poly: Polygon,
) -> None:
    """
    Local per-door midline gap-close between rooms `ra` and `rb` (mutates their
    polygons in place). A thin patch is unioned into each host room out to the
    gap midline so the two rooms come to share a short real boundary segment
    across the opening. Each patch has ALL existing room area subtracted first,
    so it can only fill previously-unoccupied wall-gap space and can never create
    a new room overlap.

    No-op if the rooms are more than `_MAX_GAP_CLOSE_M` apart at the connector (a
    wide open passage, not a wall) — pass 2 then just snaps the endpoints onto
    `ra`'s edge, and egress loses that one door-crossing for `rb`.
    """
    _, width, _ = _rect_dims(conn)
    c = conn.centroid
    pa = nearest_points(ra["poly"].exterior, c)[0]
    pb = nearest_points(rb["poly"].exterior, c)[0]
    gap = math.hypot(pb.x - pa.x, pb.y - pa.y)
    if gap > _MAX_GAP_CLOSE_M or gap < 1e-9:
        return

    half = width / 2.0
    mx, my = (pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0
    nrm = ((pb.x - pa.x) / gap, (pb.y - pa.y) / gap)   # unit normal, points ra → rb
    perp = (-nrm[1], nrm[0])                           # unit vector along the wall
    depth = min(gap / 2.0 + _GAP_CLOSE_MARGIN_M, _GAP_CLOSE_MAX_DEPTH_M)
    ext = half + _GAP_CLOSE_MARGIN_M                   # patch over-runs the opening span

    occupied = unary_union([r["poly"] for r in rooms])
    # Permissive clip: allow patches up to _MAX_GAP_CLOSE_M outside the inner
    # shell so balcony gaps (balconies protrude past `inner`) still close, while
    # still bounding a mis-resolved door's patch to the plan's vicinity.
    clip = boundary_poly.buffer(_MAX_GAP_CLOSE_M)

    def _grow(room: dict, direction: tuple[float, float]) -> None:
        q = Polygon([
            (mx - ext * perp[0], my - ext * perp[1]),
            (mx + ext * perp[0], my + ext * perp[1]),
            (mx + ext * perp[0] + depth * direction[0], my + ext * perp[1] + depth * direction[1]),
            (mx - ext * perp[0] + depth * direction[0], my - ext * perp[1] + depth * direction[1]),
        ])
        if not q.is_valid:
            q = q.buffer(0)
        q = q.intersection(clip).difference(occupied)   # fill unoccupied gap space only
        if q.is_empty or q.area < 1e-9:
            return
        merged = unary_union([room["poly"], q])
        if isinstance(merged, MultiPolygon):
            merged = max(merged.geoms, key=lambda g: g.area)
        if isinstance(merged, Polygon) and merged.is_valid and not merged.is_empty:
            room["poly"] = Polygon(merged.exterior)

    _grow(ra, (-nrm[0], -nrm[1]))   # patch reaches back into ra
    _grow(rb, (nrm[0], nrm[1]))     # patch reaches back into rb


def _opening(oid: str, otype: str, host_wall: list[str], e1, e2, clear_width: float) -> dict:
    return {
        "id": oid,
        "type": otype,
        "host_wall": host_wall,
        "endpoints": [list(e1), list(e2)],
        "clear_width": clear_width,
        "swing": "inward",
    }

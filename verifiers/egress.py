"""
Egress travel-distance verifier.

PLACEHOLDER THRESHOLD: 30.0 m maximum egress travel distance.
PLACEHOLDER — generic/placeholder value, not a real statutory jurisdiction.
Results must not be read as evidence of real-world building-code compliance.
See LIMITATIONS.md.

Measures the shortest walkable-path distance from the building entrance to the
most remote polygon vertex in each room.

Reference point (what we measure FROM): all polygon vertices of each room;
the maximum shortest-path distance over all vertices is the room's egress
distance.  For polygonal rooms the worst-case point is always at a vertex under
shortest-path distance, so this is exact for convex rooms.

Graph construction:
  - Each room vertex and door midpoint is a room-specific node (keyed by
    (room_id, vertex_index) or (door_id, room_id)).  This prevents shared
    polygon corners from acting as inter-room passage bridges — only door
    midpoints serve as gateways between rooms.
  - Within-room edges are added only when the straight segment between two
    local nodes lies entirely within the room polygon (Shapely covers() filter).
    This blocks physically invalid shortcuts through concave notches.
  - Dijkstra from the entrance node yields shortest distances to all vertices.

Residual approximation: within a single room, travel is modelled as straight-
line segments between polygon vertices and door midpoints.  If a room is
concave and the covers() filter disconnects some vertex pairs from the door
midpoint, those vertices may report as unreachable even though a valid in-room
path exists.  This is a known limitation of the waypoint-graph model; see
LIMITATIONS.md and DECISIONS.md 2026-08-26.

Margin sign convention (maximum-type clause, per PROJECT_CONTEXT.md reward spec):
    margin = (threshold − measured) / max(|threshold|, ε)
Positive margin = compliant; negative = violation.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

from shapely.geometry import LineString, Polygon

# PLACEHOLDER — generic threshold, not a real jurisdiction (see LIMITATIONS.md)
_THRESHOLD_M: float = 30.0
_CLAUSE_ID: str = "EGR-D-01"
_EPS: float = 1e-9

# Synthetic entrance node key (not a coordinate tuple — prevents type collisions)
_ENTRANCE = "entrance"


def check(plan: dict) -> list[dict]:
    """
    Check every room against the maximum egress travel-distance threshold.

    Returns
    -------
    list of audit records, one per room:
        clause_id      : str
        room_id        : str
        measured_value : float  (metres; absent on ERROR)
        threshold      : float  (metres; absent on ERROR)
        margin         : float  (absent on ERROR)
        status         : "PASS" | "FAIL" | "ERROR"
        detail         : str    (present only on ERROR)
    """
    adj = _build_graph(plan)
    dist = _dijkstra(adj, _ENTRANCE)

    results = []
    for room in plan.get("rooms", []):
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

        vertex_keys = [("rv", rid, i) for i in range(len(raw))]
        vertex_dists = [dist.get(k, float("inf")) for k in vertex_keys]

        if any(d == float("inf") for d in vertex_dists):
            results.append({
                "clause_id": _CLAUSE_ID,
                "room_id": rid,
                "status": "ERROR",
                "detail": (
                    "one or more room vertices not reachable from entrance — "
                    "run topology verifier first, or check within-room concavity"
                ),
            })
            continue

        max_dist = max(vertex_dists)
        margin = round((_THRESHOLD_M - max_dist) / max(abs(_THRESHOLD_M), _EPS), 6)
        results.append({
            "clause_id": _CLAUSE_ID,
            "room_id": rid,
            "measured_value": round(max_dist, 4),
            "threshold": _THRESHOLD_M,
            "margin": margin,
            "status": "PASS" if margin >= 0 else "FAIL",
        })

    return results


# ── Graph construction ────────────────────────────────────────────────────────

def _build_graph(plan: dict) -> dict:
    """
    Build the egress waypoint graph and return it as an adjacency dict.

    Node key scheme
    ---------------
    ("rv", room_id, vertex_index)  — room vertex (source for egress measurement)
    ("dm", door_id, room_id)       — door midpoint in context of one room
                                     (the same physical point has two keys, one
                                     per room it separates; they are 0-cost linked)
    "entrance"                     — building entrance point (Dijkstra source)

    Inter-room travel is ONLY possible through door midpoints.  Shared polygon
    vertices at room corners do NOT create bridges between rooms.
    """
    adj: dict = defaultdict(list)

    def _edge(a, b, w: float) -> None:
        adj[a].append((b, w))
        adj[b].append((a, w))

    entrance_coord = tuple(
        map(float, plan.get("boundary", {}).get("entrance", [0.0, 0.0]))
    )

    # Pre-compute door midpoints (only for type "door" openings with 2 endpoints)
    door_mids: dict[str, tuple] = {}
    for o in plan.get("openings", []):
        if o.get("type") != "door":
            continue
        ep = o.get("endpoints", [])
        if len(ep) == 2:
            door_mids[o["id"]] = (
                (ep[0][0] + ep[1][0]) / 2.0,
                (ep[0][1] + ep[1][1]) / 2.0,
            )

    # Entrance → exterior door midpoints
    for o in plan.get("openings", []):
        if o.get("type") != "door":
            continue
        hw = o.get("host_wall", [])
        if "exterior" not in hw:
            continue
        did = o["id"]
        mid = door_mids.get(did)
        if mid is None:
            continue
        for rid in hw:
            if rid == "exterior":
                continue
            _edge(_ENTRANCE, ("dm", did, rid), math.dist(entrance_coord, mid))

    # Door crossing edges (0 cost between the two room sides of each interior door)
    for o in plan.get("openings", []):
        if o.get("type") != "door":
            continue
        did = o["id"]
        if did not in door_mids:
            continue
        interior_rooms = [h for h in o.get("host_wall", []) if h != "exterior"]
        if len(interior_rooms) == 2:
            ra, rb = interior_rooms
            _edge(("dm", did, ra), ("dm", did, rb), 0.0)

    # Within-room edges: polygon vertices + door midpoints, Shapely covers() filter
    for room in plan.get("rooms", []):
        rid = room["id"]
        raw = room.get("polygon", [])
        if len(raw) < 3:
            continue
        verts = [tuple(map(float, p)) for p in raw]
        poly = Polygon(verts)
        if not poly.is_valid or poly.area < _EPS:
            continue

        local: list[tuple] = []  # (node_key, (x, y))
        for i, v in enumerate(verts):
            local.append((("rv", rid, i), v))
        for o in plan.get("openings", []):
            if o.get("type") != "door":
                continue
            if rid not in o.get("host_wall", []):
                continue
            did = o["id"]
            mid = door_mids.get(did)
            if mid is not None:
                local.append((("dm", did, rid), mid))

        for j in range(len(local)):
            key_a, ca = local[j]
            for k in range(j + 1, len(local)):
                key_b, cb = local[k]
                if ca == cb:
                    continue
                seg = LineString([ca, cb])
                try:
                    inside = poly.covers(seg)
                except Exception:
                    inside = False
                if inside:
                    _edge(key_a, key_b, math.dist(ca, cb))

    return dict(adj)


def _dijkstra(adj: dict, source) -> dict:
    """
    Dijkstra with an integer tiebreaker — avoids comparing heterogeneous node
    key types (strings vs tuples) inside the heap.
    """
    dist: dict = {source: 0.0}
    counter = 0
    heap = [(0.0, counter, source)]
    while heap:
        d, _, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                counter += 1
                heapq.heappush(heap, (nd, counter, v))
    return dist

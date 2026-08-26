# Decisions

Dated log of every design choice and the reasoning behind it.

---

## 2026-08-26 — Day 1: Verifier bank v1

### Geometry library: Shapely
Approved as the sole geometry dependency for all verifiers (and Day 2 egress).
Shapely ≥ 1.7 is pre-installed on Colab; locally installed via pip. Using Shapely 2.1.2.
Rationale: handles intersection, validity, and polygon topology correctly with no
hand-rolled geometry. Rolling our own would be error-prone on the edge cases
(collinear vertices, shared-edge artefacts, concave polygons).

### Placeholder thresholds
- **Corridor minimum clear width: 0.90 m** (PLACEHOLDER — generic code, not a real jurisdiction)
- **Door minimum clear width: 0.80 m** (PLACEHOLDER — generic code, not a real jurisdiction)
These values were chosen as round numbers in the plausible range for residential corridors
and doors in many informal guidance sources. They are not taken from any specific code body.
Label them as placeholder in every code comment, docstring, and doc — see LIMITATIONS.md.

### Corridor type aliases
Exactly `"corridor"` only. Aliases (`"hallway"`, `"hall"`, `"passage"`, `"circulation"`)
deferred until Day 3, when real RPLAN type labels are known. Guessing aliases now would
risk silent false negatives if the label set differs from expectations.

### host_wall schema (previously undocumented)
An opening's `host_wall` field is an array of 1–2 room IDs the wall separates:
- Interior wall: `["room_a", "room_b"]`
- Exterior wall: `["exterior", "room_x"]`
`"exterior"` is a reserved pseudo-ID resolved to the building boundary polygon's edges.
This was agreed verbally and is now the canonical schema definition.

### Corridor clear-width algorithm: per-edge normal chord sampling
For each polygon edge, cast 10 cross-section chords perpendicular to that edge
(using the edge's inward normal direction). The minimum chord segment length across
all edges and sample positions is the reported clear width.

Rejected alternative — single global MBR axis: an L-shaped corridor's turned arm
runs roughly parallel to the MBR major axis, so MBR-axis cuts measure that arm's
length rather than its width. Per-edge normals solve this because each arm contributes
its own edges whose normals naturally sample across that arm.

Shapely normalises all Polygon exterior rings to CCW orientation, so the inward normal
is always the left-hand perpendicular of each directed edge — no per-polygon orientation
check required.

---

## 2026-08-26 — Day 2: Topology + Egress verifiers

### Placeholder threshold: egress maximum travel distance
- **Egress maximum travel distance: 30.0 m** (PLACEHOLDER — generic code, not a real jurisdiction)
Round number plausible for residential egress. Not taken from any code body.
Clause ID: EGR-D-01.

### Egress reference point: polygon vertices, take max
The "egress distance" of a room is defined as the maximum shortest-path distance
from the building entrance to any polygon vertex of that room. All vertices are
measured, and the worst-case is reported. For convex rooms this is exact; for
concave rooms it is an approximation (the worst-case point may not be at a vertex
if the room has significant concavity away from vertices). See LIMITATIONS.md.

### Egress graph: room-id-based node keys
Within-room graph nodes are keyed by `("rv", room_id, vertex_index)` for room
vertices and `("dm", door_id, room_id)` for door midpoints. This prevents shared
polygon corners (e.g., a wall corner that two rooms share) from acting as implicit
inter-room bridges. Only door midpoints serve as gateways between rooms.

Rejected alternative — raw coordinate tuples as keys: two rooms sharing a wall
corner would produce the same coordinate key, creating an implicit passage without
a door. Room-id keys make crossing explicit.

### Within-room visibility: Shapely covers() filter
A within-room edge between two local nodes (polygon vertices or door midpoints) is
only added to the graph when `poly.covers(LineString([a, b]))` is True — i.e., the
entire segment lies within (or on the boundary of) the room polygon. This blocks
shortcuts through concave notches in L-shaped or non-convex rooms.

Residual approximation: within-room paths are modelled as straight-line segments
between waypoints. For rooms with significant concavity, some vertex pairs may not
be directly visible from the door midpoint even along the polygon boundary; those
vertices would be reported as unreachable. See LIMITATIONS.md.

### Dijkstra with integer tiebreaker
Python's `heapq` cannot compare tuples against strings (the entrance node is the
string `"entrance"`; room-vertex nodes are tuples). A monotonically-increasing
integer counter is inserted as the second heap element to break distance ties
without ever comparing node keys directly.

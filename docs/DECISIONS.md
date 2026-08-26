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

---

## 2026-08-26 — Reward function skeleton (Day 6-7 work, pulled forward)

Built ahead of schedule while waiting on RPLAN data access. Does not depend on the
dataset or the trained model — only on the verifier bank (geometry_gate, topology,
corridor, door, egress), which is complete and fully tested. Day 6-7 will pick up
the GRPO training loop; the reward function itself will not need to be rebuilt then.

### Hard gate scope: geometry_gate + topology + no_rooms
`geometry_gate` and `topology` are both labeled "structural gates, not statutory
clauses" in their code. Both trigger R(y) = −1 for the same reason: statutory clause
measurements (corridor widths, egress distances) would be unreliable or return ERROR
records if the plan's structural invariants are not satisfied first.
`topology.check` passing is a pre-condition for `egress.check` producing numeric
results — an isolated room produces an egress ERROR, not a margin.
The `no_rooms` guard catches the schema gap where `geometry_gate` silently passes on
an empty plan (it calls `.get("rooms", [])` internally).

### Schema gap: no JSON syntax/schema validator
PROJECT_CONTEXT.md's staged verifier bank lists a "syntax/schema gate" before
geometry_gate, but no such module was built in Days 1-2.  `geometry_gate` covers
geometric validity but not schema validity (e.g., missing required keys, wrong types
on non-geometry fields, missing `clear_width` on a door).
Decision: defer a full schema validator to Day 3/4 when real RPLAN JSON is in hand.
Minimal guard added to `compute_reward`: if `plan.get("rooms")` is falsy (empty list
or missing key), return the hard-gate penalty with reason "no_rooms".

### ERROR records: treated as FAIL with synthetic margin = −1.0
An ERROR record means the verifier could not produce a numeric measurement.
Post-gate ERRORs most commonly arise from:
  - `door.check`: missing or non-numeric `clear_width` field (schema gap)
  - `egress.check`: within-room waypoint graph disconnected by extreme concavity
Treating ERROR as FAIL with the maximum possible severity (margin = −1.0) ensures
the reward signal pushes away from plans that break any verifier.  The synthetic
margin −1.0 is documented in `reward.py`'s module docstring (not only here) because
it is a non-obvious design choice affecting every ERROR that reaches `R_code`.

### Clause aggregation: worst-case margin per clause
Each statutory clause (CORR-W-01, DOOR-W-01, EGR-D-01) may produce multiple audit
records.  The clause margin m_k is the minimum margin across all its records; the
severity s_k is `max(0, −min_margin)`.  Rationale: building code requires every
instance to comply — one non-compliant door fails the door clause.

### Vacuous clause pass for missing records
If a plan has no rooms of the relevant type (e.g., no corridor-typed rooms),
`corridor.check` returns `[]` and CORR-W-01 has no records.  Treated as c_k=1,
s_k=0 — the clause weight still contributes to R_code.  Rationale: the project
scope does not mandate corridors; the egress and topology checks are sufficient to
ensure traversability without them.

### Training constants (PLACEHOLDER/tunable — not statutory thresholds)
These are training hyperparameters.  They are labeled placeholder/tunable because
they have not been calibrated on real training runs; expect at least one tuning pass
during Day 6-7.

- **w_k = 1/3 per clause** — equal weights, normalised to sum to 1 over 3 clauses.
  Simplest starting point; no prior information about which clause matters most.

- **λ = 0.5** — severity penalty at half the per-clause weight.  Violations are
  penalised but not catastrophically; a slightly non-compliant plan can still have
  a positive R_code (which matters for GRPO group variance).

- **M = 2.0** — compliance domination bonus.  Chosen so that the worst possible
  compliant reward (M + 1.0 + 0 = 3.0) strictly exceeds the best possible
  statutory-violating reward (0 + 2/3 + β = 0.767).  Algebraic requirement:
  M > β − 1/3; M = 2.0 gives a factor-of-4 margin above the minimum required value.

- **β = 0.1** — quality score coefficient.  Q(y) ∈ (0,1], so β*Q ≤ 0.1.  Keeps
  quality from overriding even a small compliance gap while still differentiating
  plans within the compliant region.

### Q(y): area-weighted mean isoperimetric compactness (PLACEHOLDER)
`compactness_k = 4π * area_k / perimeter_k²`  (range (0,1]; circle=1, square≈0.785)
`Q(y) = Σ(area_k * compactness_k) / Σ(area_k)`

Area-weighted so large habitable rooms (living room, bedroom) dominate; a small
misshapen closet doesn't swamp the score.
Averaging choice: area-weighted (not unweighted, not over the boundary polygon).
  - Unweighted mean rejected: a tiny well-shaped room can pad the score and can be
    gamed by adding many small square rooms.
  - Boundary polygon rejected: measures floor-plate shape, not room quality.
PLACEHOLDER — this is a differentiating signal within the compliant region only.
Revisit once training reveals whether the signal is useful or noisy.

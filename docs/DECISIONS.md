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

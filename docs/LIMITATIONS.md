# Limitations

Clause thresholds are placeholder/generic code, not a real statutory jurisdiction — results are not evidence of real-world compliance.

## Within-room convexity approximation (egress.py)

The egress verifier models within-room travel as straight-line segments between
polygon vertices and door midpoints. Each candidate segment is accepted only when
`Shapely.covers(LineString)` confirms the segment lies entirely inside the room
polygon. This is exact for convex rooms.

For concave rooms (L-shaped, notched, or otherwise non-convex):
- The covers() filter correctly blocks segments that would cut through a notch.
- However, some vertex-to-door-midpoint paths may be blocked even though a valid
  in-room path exists — the path would need to bend around the concavity, but the
  waypoint graph can only travel in straight lines between its fixed waypoints.
- In practice this means the measured egress distance for a heavily concave room
  may be reported as an ERROR ("vertex not reachable") even when it is physically
  reachable. The workaround is to add intermediate waypoints at concavity junctions,
  which is not implemented in v1.

The `test_concave_room_covers_filter` test exercises the L-shape case and confirms
the detour path is taken (measured > 13.5 m vs. straight-line ≈ 11.18 m). Rooms
with severe concavity beyond this L-shape approximation should be validated manually.

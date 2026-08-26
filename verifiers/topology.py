"""
Topology verifier — door-graph reachability.

Checks that every room can reach the building entrance through door openings only
(windows do not count as passage routes).

This is a structural gate, not a statutory clause.  It returns pass/fail plus
which rooms are unreachable — there is no threshold record shape.

Graph: nodes = room IDs + synthetic "entrance" node.
Edges: one per door opening, connecting the two IDs in host_wall
  ("exterior" → treated as the "entrance" node).
BFS from "entrance"; unreachable rooms are reported.
"""

from __future__ import annotations

from collections import defaultdict, deque

_ENTRANCE = "entrance"  # synthetic node for the building entry point


def check(plan: dict) -> dict:
    """
    Parameters
    ----------
    plan : dict
        Floor-plan JSON conforming to the project schema.

    Returns
    -------
    dict:
        "pass"             : bool
        "unreachable_rooms": list[str]  (sorted; empty on full PASS)
    """
    adj: dict[str, set[str]] = defaultdict(set)

    for opening in plan.get("openings", []):
        if opening.get("type") != "door":
            continue
        hw = opening.get("host_wall", [])
        # Normalise: "exterior" → _ENTRANCE; anything else → room ID
        nodes = [_ENTRANCE if h == "exterior" else h for h in hw]
        if len(nodes) == 2:
            adj[nodes[0]].add(nodes[1])
            adj[nodes[1]].add(nodes[0])
        # Single-ID host_wall is malformed (geometry_gate reports it first); skip.

    room_ids = [r["id"] for r in plan.get("rooms", [])]

    # BFS from the entrance node
    visited: set[str] = {_ENTRANCE}
    queue: deque[str] = deque([_ENTRANCE])
    while queue:
        node = queue.popleft()
        for nbr in adj[node]:
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

    unreachable = sorted(rid for rid in room_ids if rid not in visited)
    return {"pass": len(unreachable) == 0, "unreachable_rooms": unreachable}

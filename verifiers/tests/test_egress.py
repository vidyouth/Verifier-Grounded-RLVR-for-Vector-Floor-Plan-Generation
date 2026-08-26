"""Tests for verifiers.egress.

PLACEHOLDER THRESHOLD used throughout: 30.0 m maximum travel distance.
Results are not evidence of real-world compliance — see LIMITATIONS.md.
"""

import math
import pytest
from verifiers.egress import check


# ── Shared simple plan builders ───────────────────────────────────────────────

def _simple_plan(room_length: float, entrance_x: float = 0.0) -> dict:
    """Single rectangular room, width=2m, entrance on left wall."""
    return {
        "boundary": {
            "polygon": [[0, 0], [room_length, 0], [room_length, 2], [0, 2]],
            "entrance": [entrance_x, 1.0],
        },
        "rooms": [
            {"id": "r1", "type": "living_room",
             "polygon": [[0, 0], [room_length, 0], [room_length, 2], [0, 2]]}
        ],
        "openings": [
            {"id": "d0", "type": "door", "host_wall": ["exterior", "r1"],
             "endpoints": [[0, 0], [0, 2]], "clear_width": 0.9},
        ],
    }


# ── Canonical PASS ────────────────────────────────────────────────────────────

def test_canonical_pass():
    """5 m room, entrance at left wall: max dist ≈ 5.1 m well under 30 m."""
    plan = _simple_plan(5.0)
    results = check(plan)
    assert len(results) == 1
    r = results[0]
    assert r["status"] == "PASS"
    assert r["measured_value"] < 30.0
    assert r["margin"] > 0
    assert r["clause_id"] == "EGR-D-01"
    assert r["room_id"] == "r1"


# ── Canonical FAIL ────────────────────────────────────────────────────────────

def test_canonical_fail():
    """35 m room, entrance at left wall: far corner at ≈ 35.03 m > 30 m → FAIL."""
    plan = _simple_plan(35.0)
    results = check(plan)
    assert len(results) == 1
    r = results[0]
    assert r["status"] == "FAIL"
    assert r["measured_value"] > 30.0
    assert r["margin"] < 0


# ── Near-threshold ────────────────────────────────────────────────────────────

def test_near_threshold_pass_29m():
    """
    29 m room, entrance at (0, 0.5) matching left-wall door midpoint.
    Door midpoint at (0, 1.0); vertex (29, 0) distance ≈ sqrt(29² + 1²) ≈ 29.017 m < 30.
    """
    plan = {
        "boundary": {
            "polygon": [[0, 0], [29, 0], [29, 2], [0, 2]],
            "entrance": [0, 1.0],
        },
        "rooms": [
            {"id": "r1", "type": "living_room",
             "polygon": [[0, 0], [29, 0], [29, 2], [0, 2]]}
        ],
        "openings": [
            {"id": "d0", "type": "door", "host_wall": ["exterior", "r1"],
             "endpoints": [[0, 0], [0, 2]], "clear_width": 0.9},
        ],
    }
    results = check(plan)
    assert results[0]["status"] == "PASS"
    assert results[0]["measured_value"] < 30.0


def test_near_threshold_fail_31m():
    """
    31 m room: far vertex at ≈ 31.016 m > 30 m → FAIL.
    """
    plan = {
        "boundary": {
            "polygon": [[0, 0], [31, 0], [31, 2], [0, 2]],
            "entrance": [0, 1.0],
        },
        "rooms": [
            {"id": "r1", "type": "living_room",
             "polygon": [[0, 0], [31, 0], [31, 2], [0, 2]]}
        ],
        "openings": [
            {"id": "d0", "type": "door", "host_wall": ["exterior", "r1"],
             "endpoints": [[0, 0], [0, 2]], "clear_width": 0.9},
        ],
    }
    results = check(plan)
    assert results[0]["status"] == "FAIL"
    assert results[0]["measured_value"] > 30.0


# ── Concave room / covers() filter ───────────────────────────────────────────

# L-shaped room: outer box 10×10 with top-right 9.5×9.5 notch cut out.
# Vertices (CCW): (0,0)→(10,0)→(10,0.5)→(0.5,0.5)→(0.5,10)→(0,10)
# Thin arms: right arm 0→10 wide, 0.5 tall; top arm 0.5 wide, 0→10 tall.
# Exterior door midpoint: (0, 5.0).
# Without covers() filter the shortest in-room segment from (0,5)→(10,0) ≈ 11.18 m
#   would be admitted even though it exits the L-shape.
# With filter: path must go corner (0,0.5)→(0.5,0.5)→(0.5,0) or similar detour;
#   the measured distance to the far corner (10,0) should exceed 13.5 m.

_CONCAVE_ROOM_PLAN = {
    "boundary": {
        "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "entrance": [0, 5.0],
    },
    "rooms": [
        {
            "id": "r1",
            "type": "bedroom",
            "polygon": [
                [0, 0], [10, 0], [10, 0.5], [0.5, 0.5],
                [0.5, 10], [0, 10],
            ],
        }
    ],
    "openings": [
        {
            "id": "d0",
            "type": "door",
            "host_wall": ["exterior", "r1"],
            "endpoints": [[0, 4.5], [0, 5.5]],
            "clear_width": 1.0,
        }
    ],
}


def test_concave_room_covers_filter():
    """
    L-shaped room: covers() filter blocks the direct (0,5)→(10,0) diagonal
    (≈ 11.18 m) that crosses through the notch.  True in-room path detours via
    the corner, making the measured distance exceed 13.5 m.
    """
    results = check(_CONCAVE_ROOM_PLAN)
    assert len(results) == 1
    r = results[0]
    # Must not be an ERROR — the room is reachable; it just takes a detour.
    assert r["status"] in ("PASS", "FAIL"), f"unexpected status: {r['status']}: {r.get('detail')}"
    # The detour path should be substantially longer than the naive straight line.
    assert r["measured_value"] > 13.5, (
        f"expected detour path > 13.5 m but got {r['measured_value']} — "
        "covers() filter may not be blocking the concave shortcut"
    )


# ── Detour: straight-line < actual path ──────────────────────────────────────

# Room A (10×2) and room B (2×8 strip on the right), connected by door d1 at
# the shared boundary. Entrance on A's left wall.
# Straight-line from entrance to B's far corner (10, 10): ≈ 13.45 m.
# Actual walkable path: entrance(0,1) → d1_mid(9,2) + (9,2)→(10,10) ≈ 9.06+8.06 = 17.12 m.

_DETOUR_PLAN = {
    "boundary": {
        "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "entrance": [0, 1.0],
    },
    "rooms": [
        {"id": "A", "type": "living_room",
         "polygon": [[0, 0], [10, 0], [10, 2], [0, 2]]},
        {"id": "B", "type": "bedroom",
         "polygon": [[8, 2], [10, 2], [10, 10], [8, 10]]},
    ],
    "openings": [
        {"id": "d0", "type": "door", "host_wall": ["exterior", "A"],
         "endpoints": [[0, 0], [0, 2]], "clear_width": 0.9},
        {"id": "d1", "type": "door", "host_wall": ["A", "B"],
         "endpoints": [[8, 2], [10, 2]], "clear_width": 0.9},
    ],
}


def test_detour_path_longer_than_straight_line():
    """
    Room B is geometrically near (straight-line ≈ 13.45 m) but must be accessed
    via room A and then d1 (total path ≈ 17 m).  Measured distance must exceed 14 m.
    """
    results = check(_DETOUR_PLAN)
    b_result = next((r for r in results if r["room_id"] == "B"), None)
    assert b_result is not None, "no result for room B"
    assert b_result["status"] in ("PASS", "FAIL"), f"unexpected: {b_result}"
    assert b_result["measured_value"] > 14.0, (
        f"expected detour path > 14 m but got {b_result['measured_value']} — "
        "room-id node keys may be allowing a shortcut through shared corners"
    )


# ── Mutation adversarial ──────────────────────────────────────────────────────

def test_mutation_door_to_window_causes_error():
    """Exterior door mutated to window → room has no path to entrance → ERROR."""
    plan = _simple_plan(5.0)
    plan["openings"][0]["type"] = "window"
    results = check(plan)
    assert results[0]["status"] == "ERROR"


def test_mutation_pass_to_fail_by_extending_room():
    """Extend room beyond 30 m threshold → flips from PASS to FAIL."""
    short = _simple_plan(5.0)
    long_ = _simple_plan(35.0)
    assert check(short)[0]["status"] == "PASS"
    assert check(long_)[0]["status"] == "FAIL"


# ── Error cases ───────────────────────────────────────────────────────────────

def test_room_not_reachable_from_entrance_error():
    """No exterior door at all → room vertex unreachable → ERROR."""
    plan = {
        "boundary": {
            "polygon": [[0, 0], [5, 0], [5, 2], [0, 2]],
            "entrance": [0, 1.0],
        },
        "rooms": [
            {"id": "r1", "type": "bedroom",
             "polygon": [[0, 0], [5, 0], [5, 2], [0, 2]]}
        ],
        "openings": [],  # no doors at all
    }
    results = check(plan)
    assert results[0]["status"] == "ERROR"


def test_degenerate_polygon_error():
    """Room with only 2 polygon vertices → ERROR."""
    plan = {
        "boundary": {"polygon": [[0, 0], [5, 0], [5, 2], [0, 2]], "entrance": [0, 1]},
        "rooms": [{"id": "r1", "type": "bedroom", "polygon": [[0, 0], [5, 0]]}],
        "openings": [
            {"id": "d0", "type": "door", "host_wall": ["exterior", "r1"],
             "endpoints": [[0, 0], [0, 2]], "clear_width": 0.9}
        ],
    }
    results = check(plan)
    assert results[0]["status"] == "ERROR"


# ── Multiple rooms ────────────────────────────────────────────────────────────

def test_multiple_rooms_each_checked():
    """
    Three-room chain: entrance→r1→r2→r3 via doors.
    All rooms close to entrance → all PASS.
    """
    plan = {
        "boundary": {
            "polygon": [[0, 0], [12, 0], [12, 2], [0, 2]],
            "entrance": [0, 1.0],
        },
        "rooms": [
            {"id": "r1", "type": "living_room",
             "polygon": [[0, 0], [4, 0], [4, 2], [0, 2]]},
            {"id": "r2", "type": "bedroom",
             "polygon": [[4, 0], [8, 0], [8, 2], [4, 2]]},
            {"id": "r3", "type": "bedroom",
             "polygon": [[8, 0], [12, 0], [12, 2], [8, 2]]},
        ],
        "openings": [
            {"id": "d0", "type": "door", "host_wall": ["exterior", "r1"],
             "endpoints": [[0, 0], [0, 2]], "clear_width": 0.9},
            {"id": "d1", "type": "door", "host_wall": ["r1", "r2"],
             "endpoints": [[4, 0], [4, 2]], "clear_width": 0.9},
            {"id": "d2", "type": "door", "host_wall": ["r2", "r3"],
             "endpoints": [[8, 0], [8, 2]], "clear_width": 0.9},
        ],
    }
    results = check(plan)
    assert len(results) == 3
    ids = {r["room_id"] for r in results}
    assert ids == {"r1", "r2", "r3"}
    for r in results:
        assert r["status"] == "PASS", f"room {r['room_id']} failed: {r}"


def test_empty_rooms_list():
    """No rooms → empty result list."""
    plan = {
        "boundary": {"polygon": [[0, 0], [5, 0], [5, 2], [0, 2]], "entrance": [0, 1]},
        "rooms": [],
        "openings": [],
    }
    assert check(plan) == []

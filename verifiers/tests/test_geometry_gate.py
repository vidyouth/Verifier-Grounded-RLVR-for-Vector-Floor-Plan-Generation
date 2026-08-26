"""Tests for verifiers.geometry_gate."""

import copy
import pytest
from verifiers.geometry_gate import check


# ── Baseline valid plan ───────────────────────────────────────────────────────
#
#  Boundary: 6 m × 5 m
#  r1 (corridor): 6 m × 1.2 m strip at bottom
#  r2 (bedroom):  6 m × 3.8 m above
#  o1 (door): on shared wall y=1.2, clear span x=1.0 to x=1.8

_VALID_PLAN: dict = {
    "metadata": {
        "schema_version": "0.1",
        "jurisdiction": "placeholder/generic — not a real jurisdiction",
    },
    "boundary": {
        "polygon": [[0, 0], [6, 0], [6, 5], [0, 5]],
        "entrance": [3, 0],
    },
    "rooms": [
        {
            "id": "r1",
            "type": "corridor",
            "polygon": [[0, 0], [6, 0], [6, 1.2], [0, 1.2]],
        },
        {
            "id": "r2",
            "type": "bedroom",
            "polygon": [[0, 1.2], [6, 1.2], [6, 5], [0, 5]],
        },
    ],
    "openings": [
        {
            "id": "o1",
            "type": "door",
            "host_wall": ["r1", "r2"],
            "endpoints": [[1.0, 1.2], [1.8, 1.2]],
            "clear_width": 0.8,
            "swing": "inward",
        }
    ],
}


def _mut(plan: dict, **overrides) -> dict:
    """Deep copy of plan with top-level key overrides."""
    p = copy.deepcopy(plan)
    p.update(overrides)
    return p


# ── Canonical cases ───────────────────────────────────────────────────────────

def test_canonical_pass():
    result = check(_VALID_PLAN)
    assert result["pass"] is True
    assert result["issues"] == []


def test_canonical_fail_overlapping_rooms():
    """r2 bottom edge moved to y=0.5 — overlaps r1 by a 0.7 m strip."""
    plan = _mut(_VALID_PLAN, rooms=[
        {"id": "r1", "type": "corridor", "polygon": [[0, 0], [6, 0], [6, 1.2], [0, 1.2]]},
        {"id": "r2", "type": "bedroom",  "polygon": [[0, 0.5], [6, 0.5], [6, 5], [0, 5]]},
    ])
    result = check(plan)
    assert result["pass"] is False
    assert any(i["code"] == "ROOM_OVERLAP" for i in result["issues"])


# ── Polygon self-intersection ─────────────────────────────────────────────────

def test_self_intersecting_polygon():
    """Bow-tie shape: (0,0)→(1,1)→(1,0)→(0,1) — edges cross."""
    plan = _mut(_VALID_PLAN, rooms=[
        {"id": "r1", "type": "corridor", "polygon": [[0, 0], [1, 1], [1, 0], [0, 1]]},
    ])
    result = check(plan)
    assert result["pass"] is False
    codes = {i["code"] for i in result["issues"]}
    assert "ROOM_SELF_INTERSECT" in codes or "ROOM_INVALID" in codes


# ── Opening placement ─────────────────────────────────────────────────────────

def test_opening_on_shared_wall_pass():
    """Door endpoints exactly on y=1.2 shared wall — valid plan passes."""
    assert check(_VALID_PLAN)["pass"] is True


def test_opening_off_wall_fail():
    """Door endpoints moved to y=1.5 — not on any host_wall edge."""
    plan = copy.deepcopy(_VALID_PLAN)
    plan["openings"][0]["endpoints"] = [[1.0, 1.5], [1.8, 1.5]]
    result = check(plan)
    assert result["pass"] is False
    assert any(i["code"] == "OPENING_OFF_WALL" for i in result["issues"])


def test_exterior_door_pass():
    """Door on boundary (host_wall=['exterior','r1']) with endpoints on boundary edge."""
    plan = copy.deepcopy(_VALID_PLAN)
    plan["openings"] = [{
        "id": "o_ext",
        "type": "door",
        "host_wall": ["exterior", "r1"],
        "endpoints": [[1.0, 0.0], [1.8, 0.0]],
        "clear_width": 0.9,
        "swing": "outward",
    }]
    assert check(plan)["pass"] is True


def test_unresolved_host_wall():
    """host_wall references a room ID that doesn't exist in the plan."""
    plan = copy.deepcopy(_VALID_PLAN)
    plan["openings"][0]["host_wall"] = ["nonexistent"]
    result = check(plan)
    assert result["pass"] is False
    assert any(i["code"] == "OPENING_UNRESOLVED_HOST" for i in result["issues"])


def test_opening_bad_endpoint_count():
    """Opening with only one endpoint is malformed."""
    plan = copy.deepcopy(_VALID_PLAN)
    plan["openings"][0]["endpoints"] = [[1.0, 1.2]]
    result = check(plan)
    assert result["pass"] is False
    assert any(i["code"] == "OPENING_BAD_ENDPOINTS" for i in result["issues"])


# ── Near-threshold edge cases ─────────────────────────────────────────────────

def test_rooms_touching_at_edge_pass():
    """
    Near-threshold pass: r1 top edge == r2 bottom edge at y=1.2 exactly.
    Intersection area is zero (a shared line) — must not be reported as overlap.
    """
    assert check(_VALID_PLAN)["pass"] is True


def test_epsilon_overlap_fail():
    """
    Near-threshold fail: r2 bottom edge shifted 0.001 m down to y=1.199,
    creating a tiny but real interior overlap (area > 1e-9 m²).
    """
    plan = copy.deepcopy(_VALID_PLAN)
    plan["rooms"][1]["polygon"] = [[0, 1.199], [6, 1.199], [6, 5], [0, 5]]
    result = check(plan)
    assert result["pass"] is False
    assert any(i["code"] == "ROOM_OVERLAP" for i in result["issues"])


# ── Mutation-based adversarial ────────────────────────────────────────────────

def test_mutation_vertex_nudge_causes_overlap():
    """
    Start from valid plan; nudge r1's top-right corner from (6,1.2) to (6,1.8).
    r1 is now a trapezoid that overlaps r2's rectangle.
    """
    plan = copy.deepcopy(_VALID_PLAN)
    plan["rooms"][0]["polygon"] = [[0, 0], [6, 0], [6, 1.8], [0, 1.2]]
    result = check(plan)
    assert result["pass"] is False
    assert any(i["code"] == "ROOM_OVERLAP" for i in result["issues"])


# ── Degenerate inputs ─────────────────────────────────────────────────────────

def test_room_with_too_few_vertices():
    plan = _mut(_VALID_PLAN, rooms=[
        {"id": "r1", "type": "corridor", "polygon": [[0, 0], [1, 0]]},
    ])
    result = check(plan)
    assert result["pass"] is False
    assert any(i["code"] == "ROOM_DEGENERATE" for i in result["issues"])


def test_empty_plan_passes():
    """Plan with no rooms or openings has nothing to fail."""
    result = check({"rooms": [], "openings": [], "boundary": {"polygon": [[0,0],[10,0],[10,10],[0,10]]}})
    assert result["pass"] is True

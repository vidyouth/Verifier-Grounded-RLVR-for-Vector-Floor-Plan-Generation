"""Tests for verifiers.topology."""

import copy
import pytest
from verifiers.topology import check


# ── Shared fixtures ───────────────────────────────────────────────────────────
#
# Three-room linear chain:  entrance → r1 → r2 → r3
#
# Boundary: 12 m × 4 m
# r1: x 0–4,  r2: x 4–8,  r3: x 8–12
# d0: exterior door on r1's left wall
# d1: interior door r1 ↔ r2
# d2: interior door r2 ↔ r3

def _door(oid, host_wall, **kw):
    return {"id": oid, "type": "door", "host_wall": host_wall, **kw}

def _window(oid, host_wall, **kw):
    return {"id": oid, "type": "window", "host_wall": host_wall, **kw}

def _room(rid, **kw):
    return {"id": rid, **kw}

_BASE_PLAN = {
    "boundary": {"polygon": [[0,0],[12,0],[12,4],[0,4]], "entrance": [0,2]},
    "rooms": [
        _room("r1"), _room("r2"), _room("r3"),
    ],
    "openings": [
        _door("d0", ["exterior", "r1"]),
        _door("d1", ["r1", "r2"]),
        _door("d2", ["r2", "r3"]),
    ],
}


def _mut(plan, **overrides):
    p = copy.deepcopy(plan)
    p.update(overrides)
    return p


# ── Canonical cases ───────────────────────────────────────────────────────────

def test_canonical_pass_all_reachable():
    """Linear chain: entrance → r1 → r2 → r3, all via doors → PASS."""
    result = check(_BASE_PLAN)
    assert result["pass"] is True
    assert result["unreachable_rooms"] == []


def test_canonical_fail_isolated_room():
    """r3 has no door connection — unreachable."""
    plan = _mut(_BASE_PLAN, openings=[
        _door("d0", ["exterior", "r1"]),
        _door("d1", ["r1", "r2"]),
        # d2 removed: r3 is now isolated
    ])
    result = check(plan)
    assert result["pass"] is False
    assert "r3" in result["unreachable_rooms"]


# ── Near-threshold structural cases ──────────────────────────────────────────

def test_single_path_still_passes():
    """
    Near-threshold pass: only one path to entrance (linear chain).
    All rooms reachable → PASS.
    """
    result = check(_BASE_PLAN)
    assert result["pass"] is True


def test_critical_door_removal_fails():
    """
    Near-threshold fail: remove d1 (the r1↔r2 bridge).
    r2 and r3 both become unreachable.
    """
    plan = _mut(_BASE_PLAN, openings=[
        _door("d0", ["exterior", "r1"]),
        # d1 removed — r2 and r3 cut off
        _door("d2", ["r2", "r3"]),
    ])
    result = check(plan)
    assert result["pass"] is False
    assert set(result["unreachable_rooms"]) == {"r2", "r3"}


# ── Mutation-based adversarial ────────────────────────────────────────────────

def test_mutation_door_to_window_disconnects_room():
    """
    Valid plan mutated: d2 type changed from 'door' to 'window'.
    r3 can only be reached through d2 → now unreachable.
    """
    plan = copy.deepcopy(_BASE_PLAN)
    assert check(plan)["pass"] is True  # baseline passes

    plan["openings"][2]["type"] = "window"
    result = check(plan)
    assert result["pass"] is False
    assert "r3" in result["unreachable_rooms"]


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_window_only_room_unreachable():
    """Room accessible only via window is unreachable."""
    plan = _mut(_BASE_PLAN, openings=[
        _door("d0", ["exterior", "r1"]),
        _door("d1", ["r1", "r2"]),
        _window("w1", ["r2", "r3"]),  # window, not a door
    ])
    result = check(plan)
    assert result["pass"] is False
    assert "r3" in result["unreachable_rooms"]


def test_no_exterior_door_all_unreachable():
    """No opening connects to 'exterior' → entrance is isolated → all rooms unreachable."""
    plan = _mut(_BASE_PLAN, openings=[
        _door("d1", ["r1", "r2"]),
        _door("d2", ["r2", "r3"]),
        # no exterior door
    ])
    result = check(plan)
    assert result["pass"] is False
    assert set(result["unreachable_rooms"]) == {"r1", "r2", "r3"}


def test_multiple_unreachable_rooms_reported():
    """Two separate rooms unreachable — both listed, sorted."""
    plan = {
        "rooms": [_room("r1"), _room("r2"), _room("r3"), _room("r4")],
        "openings": [
            _door("d0", ["exterior", "r1"]),
            # r2, r3, r4 have no connections
        ],
    }
    result = check(plan)
    assert result["pass"] is False
    assert result["unreachable_rooms"] == ["r2", "r3", "r4"]  # sorted


def test_unreachable_list_is_sorted():
    """Sorted output is deterministic regardless of dict/set iteration order."""
    plan = {
        "rooms": [_room("r3"), _room("r1"), _room("r2")],
        "openings": [],
    }
    result = check(plan)
    assert result["unreachable_rooms"] == ["r1", "r2", "r3"]


def test_empty_plan_passes():
    """No rooms → nothing to fail."""
    result = check({"rooms": [], "openings": []})
    assert result["pass"] is True
    assert result["unreachable_rooms"] == []


def test_star_topology_all_reachable():
    """Hub-and-spoke: hub room r1 connects entrance + all other rooms."""
    plan = {
        "rooms": [_room("r1"), _room("r2"), _room("r3"), _room("r4")],
        "openings": [
            _door("d0", ["exterior", "r1"]),
            _door("d1", ["r1", "r2"]),
            _door("d2", ["r1", "r3"]),
            _door("d3", ["r1", "r4"]),
        ],
    }
    result = check(plan)
    assert result["pass"] is True

"""Tests for verifiers.corridor."""

import pytest
from verifiers.corridor import check


# ── Helpers ───────────────────────────────────────────────────────────────────

def _plan(rooms: list) -> dict:
    return {"rooms": rooms}


def _corridor(rid: str, polygon: list) -> dict:
    return {"id": rid, "type": "corridor", "polygon": polygon}


def _rect(x0: float, y0: float, x1: float, y1: float) -> list:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# ── Canonical cases ───────────────────────────────────────────────────────────

def test_canonical_pass_wide_corridor():
    """1.2 m × 5 m rectangular corridor — should pass."""
    result = check(_plan([_corridor("r1", _rect(0, 0, 5, 1.2))]))[0]
    assert result["status"] == "PASS"
    assert abs(result["measured_value"] - 1.2) < 0.05


def test_canonical_fail_narrow_corridor():
    """0.7 m × 5 m rectangular corridor — should fail."""
    result = check(_plan([_corridor("r1", _rect(0, 0, 5, 0.7))]))[0]
    assert result["status"] == "FAIL"
    assert abs(result["measured_value"] - 0.7) < 0.05


# ── Near-threshold ────────────────────────────────────────────────────────────

def test_near_threshold_pass():
    """0.91 m wide — just above 0.90 m threshold."""
    assert check(_plan([_corridor("r1", _rect(0, 0, 5, 0.91))]))[0]["status"] == "PASS"


def test_near_threshold_fail():
    """0.89 m wide — just below 0.90 m threshold."""
    assert check(_plan([_corridor("r1", _rect(0, 0, 5, 0.89))]))[0]["status"] == "FAIL"


# ── L-shaped corridor ─────────────────────────────────────────────────────────

def test_l_shaped_corridor_pass():
    """
    L-shaped corridor: horizontal arm 4 m × 1.0 m, vertical arm 1.0 m × 4 m.
    Both arms are 1.0 m wide → clear width = 1.0 m > 0.90 m → PASS.

    Polygon (CCW): (0,0)→(4,0)→(4,1)→(1,1)→(1,4)→(0,4)→(0,0)

    Manual trace — per-edge normal algorithm:
      Edge (4,1)→(1,1), inward normal (0,−1): chords at x∈[1.3,3.7] span the
        horizontal arm vertically → 1.0 m each.
      Edge (1,1)→(1,4), inward normal (−1,0): chords at y∈[1.3,3.7] span the
        vertical arm horizontally → 1.0 m each.
      Other edges produce longer chords (full arm lengths); minimum wins.
    Expected result: 1.0 m (within 0.05 m floating-point tolerance).
    """
    l_shape = [[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]]
    result = check(_plan([_corridor("r1", l_shape)]))[0]
    assert result["status"] == "PASS", (
        f"Expected PASS for 1.0 m L-shaped corridor, got {result['status']}"
    )
    assert abs(result["measured_value"] - 1.0) < 0.05, (
        f"Expected ≈1.0 m, got {result['measured_value']:.4f} m"
    )


# ── Mutation-based adversarial ────────────────────────────────────────────────

def test_l_shaped_narrow_arm_fail():
    """
    Mutation adversarial: L-shaped corridor where both arms are only 0.8 m wide.

    A naive global-MBR-axis approach would measure along the long dimension and
    might fail to detect the narrow cross-section of the turned arm.
    Per-edge normals must correctly report 0.8 m → FAIL.

    Polygon: (0,0)→(4,0)→(4,0.8)→(0.8,0.8)→(0.8,4)→(0,4)→(0,0)
    """
    l_shape = [[0, 0], [4, 0], [4, 0.8], [0.8, 0.8], [0.8, 4], [0, 4]]
    result = check(_plan([_corridor("r1", l_shape)]))[0]
    assert result["status"] == "FAIL", (
        f"Expected FAIL for 0.8 m L-shaped corridor, got {result['status']}"
    )
    assert abs(result["measured_value"] - 0.8) < 0.05, (
        f"Expected ≈0.8 m, got {result['measured_value']:.4f} m"
    )


# ── Multi-room and type filtering ─────────────────────────────────────────────

def test_non_corridor_rooms_ignored():
    """Only 'corridor' type rooms are checked."""
    plan = _plan([
        _corridor("r1", _rect(0, 0, 5, 1.2)),
        {"id": "r2", "type": "bedroom", "polygon": _rect(0, 1.2, 5, 5)},
        {"id": "r3", "type": "kitchen", "polygon": _rect(5, 0, 8, 3)},
    ])
    results = check(plan)
    assert len(results) == 1
    assert results[0]["room_id"] == "r1"


def test_multiple_corridors_checked_independently():
    """All corridor rooms are checked; each gets its own record."""
    plan = _plan([
        _corridor("c1", _rect(0, 0, 5, 1.2)),   # PASS — 1.2 m wide
        _corridor("c2", _rect(0, 2, 5, 2.7)),   # FAIL — 0.7 m wide
    ])
    results = check(plan)
    assert len(results) == 2
    by_id = {r["room_id"]: r["status"] for r in results}
    assert by_id["c1"] == "PASS"
    assert by_id["c2"] == "FAIL"


# ── Error handling ────────────────────────────────────────────────────────────

def test_degenerate_polygon_returns_error():
    """Fewer than 3 vertices → ERROR record, no crash."""
    result = check(_plan([_corridor("r1", [[0, 0], [1, 0]])]))[0]
    assert result["status"] == "ERROR"
    assert "vertices" in result["detail"]


def test_empty_rooms_list():
    assert check({"rooms": []}) == []
    assert check({}) == []

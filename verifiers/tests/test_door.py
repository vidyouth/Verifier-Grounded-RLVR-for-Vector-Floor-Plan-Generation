"""Tests for verifiers.door."""

import pytest
from verifiers.door import check


# ── Helpers ───────────────────────────────────────────────────────────────────

def _plan(*openings: dict) -> dict:
    return {"openings": list(openings)}


def _door(oid: str, **kwargs) -> dict:
    return {"id": oid, "type": "door", **kwargs}


# ── Canonical cases ───────────────────────────────────────────────────────────

def test_canonical_pass():
    result = check(_plan(_door("d1", clear_width=0.90)))[0]
    assert result["status"] == "PASS"
    assert result["measured_value"] == 0.90
    assert result["margin"] == pytest.approx(0.10)


def test_canonical_fail():
    result = check(_plan(_door("d1", clear_width=0.75)))[0]
    assert result["status"] == "FAIL"
    assert result["measured_value"] == 0.75
    assert result["margin"] < 0


# ── Near-threshold ────────────────────────────────────────────────────────────

def test_near_threshold_pass():
    """0.81 m — just above 0.80 m threshold."""
    assert check(_plan(_door("d1", clear_width=0.81)))[0]["status"] == "PASS"


def test_near_threshold_fail():
    """0.79 m — just below 0.80 m threshold."""
    assert check(_plan(_door("d1", clear_width=0.79)))[0]["status"] == "FAIL"


def test_exactly_at_threshold():
    """0.80 m — exactly at threshold; margin == 0 → PASS."""
    result = check(_plan(_door("d1", clear_width=0.80)))[0]
    assert result["status"] == "PASS"
    assert result["margin"] == pytest.approx(0.0)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_missing_clear_width():
    result = check(_plan(_door("d1")))[0]
    assert result["status"] == "ERROR"
    assert "missing" in result["detail"]


def test_non_numeric_clear_width():
    result = check(_plan(_door("d1", clear_width="wide")))[0]
    assert result["status"] == "ERROR"


def test_zero_clear_width():
    result = check(_plan(_door("d1", clear_width=0.0)))[0]
    assert result["status"] == "FAIL"
    assert "zero" in result["detail"]


def test_negative_clear_width():
    result = check(_plan(_door("d1", clear_width=-0.5)))[0]
    assert result["status"] == "FAIL"


def test_window_type_ignored():
    plan = _plan(
        _door("d1", clear_width=0.90),
        {"id": "w1", "type": "window", "clear_width": 0.30},
    )
    results = check(plan)
    assert len(results) == 1
    assert results[0]["opening_id"] == "d1"


def test_empty_plan_returns_empty():
    assert check({"openings": []}) == []
    assert check({}) == []


# ── Mutation-based adversarial ────────────────────────────────────────────────

def test_mutation_pass_to_fail():
    """Valid door mutated to clear_width just below threshold must flip to FAIL."""
    base = _door("d1", clear_width=0.85)
    assert check(_plan(base))[0]["status"] == "PASS"

    mutated = dict(base, clear_width=0.79)
    assert check(_plan(mutated))[0]["status"] == "FAIL"


def test_mutation_integer_width_treated_as_float():
    """clear_width=1 (int) should be accepted as 1.0 m."""
    result = check(_plan(_door("d1", clear_width=1)))[0]
    assert result["status"] == "PASS"
    assert result["measured_value"] == 1.0

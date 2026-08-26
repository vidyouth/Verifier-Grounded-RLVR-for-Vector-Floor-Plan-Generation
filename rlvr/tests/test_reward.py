"""
Tests for rlvr.reward.compute_reward.

PLACEHOLDER thresholds in force throughout (see LIMITATIONS.md):
  Corridor minimum clear width : 0.90 m
  Door minimum clear width     : 0.80 m
  Egress maximum travel dist   : 30.0 m

PLACEHOLDER training constants (see DECISIONS.md 2026-08-26):
  w_k = 1/3,  λ = 0.5,  M = 2.0,  β = 0.1

All plans are hand-authored synthetic fixtures following the same conventions as the
verifier unit tests (canonical pass/fail, near-threshold, mutation, adversarial).

Plan geometry (all plans share the same spatial template unless otherwise noted):

  Boundary  4 m wide, height varies per plan
  r1        living_room  4 × 4 m  at y 0–4
  r_corr    corridor     4 × W m  at y 4–(4+W)   W = corridor width under test
  r2        bedroom      4 × 4 m  above r_corr

  d0  exterior door on r1's left wall (y=0–4, x=0), endpoints (0,0.5)–(0,1.5)
  d1  interior door r1 ↔ r_corr on shared wall y=4, endpoints (1.0,4)–(1.9,4)
  d2  interior door r_corr ↔ r2 on shared wall y=(4+W), endpoints (1.0,4+W)–(1.9,4+W)

  Entrance  [0, 1.0]  (at d0 midpoint)
"""

import pytest

from rlvr.reward import compute_reward

# ── Constants (mirror reward.py, used for threshold assertions) ───────────────
_M = 2.0    # compliance domination bonus
_BETA = 0.1


# ── Plan builders ─────────────────────────────────────────────────────────────

def _three_room_plan(
    corridor_width: float,
    door_clear_width: float = 0.9,
    d2_clear_width: float | None = None,  # None = omit clear_width from d2
) -> dict:
    """
    Build the standard 3-room plan (r1 / r_corr / r2) with configurable corridor
    width and door clear widths.

    d2_clear_width=None produces a plan where d2 has no clear_width field,
    causing door.check to return ERROR for that opening.
    """
    top_corr = 4.0 + corridor_width
    top_r2 = top_corr + 4.0

    d2: dict = {
        "id": "d2",
        "type": "door",
        "host_wall": ["r_corr", "r2"],
        "endpoints": [[1.0, top_corr], [1.9, top_corr]],
    }
    if d2_clear_width is not None:
        d2["clear_width"] = d2_clear_width

    return {
        "boundary": {
            "polygon": [[0, 0], [4, 0], [4, top_r2], [0, top_r2]],
            "entrance": [0, 1.0],
        },
        "rooms": [
            {"id": "r1",     "type": "living_room",
             "polygon": [[0, 0],        [4, 0],        [4, 4],        [0, 4]]},
            {"id": "r_corr", "type": "corridor",
             "polygon": [[0, 4],        [4, 4],        [4, top_corr], [0, top_corr]]},
            {"id": "r2",     "type": "bedroom",
             "polygon": [[0, top_corr], [4, top_corr], [4, top_r2],   [0, top_r2]]},
        ],
        "openings": [
            {"id": "d0", "type": "door", "host_wall": ["exterior", "r1"],
             "endpoints": [[0, 0.5], [0, 1.5]], "clear_width": door_clear_width},
            {"id": "d1", "type": "door", "host_wall": ["r1", "r_corr"],
             "endpoints": [[1.0, 4], [1.9, 4]], "clear_width": door_clear_width},
            d2,
        ],
    }


# ── Named plan fixtures ───────────────────────────────────────────────────────

# Corridor 1.2 m (> 0.90 m), all doors 0.90 m (> 0.80 m) — all clauses PASS.
COMPLIANT_PLAN = _three_room_plan(corridor_width=1.2, door_clear_width=0.9, d2_clear_width=0.9)

# Corridor 0.91 m (just above 0.90 m), all doors 0.81 m (just above 0.80 m) — all PASS.
BORDERLINE_PLAN = _three_room_plan(corridor_width=0.91, door_clear_width=0.81, d2_clear_width=0.81)

# Corridor 0.80 m (below 0.90 m threshold) — CORR-W-01 FAIL; doors and egress PASS.
SINGLE_VIOLATION_PLAN = _three_room_plan(corridor_width=0.80, door_clear_width=0.9, d2_clear_width=0.9)

# Corridor 0.80 m + all doors 0.75 m (below 0.80 m) — CORR-W-01 and DOOR-W-01 both FAIL.
MULTI_VIOLATION_PLAN = _three_room_plan(corridor_width=0.80, door_clear_width=0.75, d2_clear_width=0.75)

# Same geometry as COMPLIANT but d2 has no clear_width → door.check returns ERROR for d2.
MISSING_CLEARWIDTH_PLAN = _three_room_plan(corridor_width=1.2, door_clear_width=0.9, d2_clear_width=None)

# Two rooms, no corridor-typed room — CORR-W-01 has no records (vacuous pass).
NO_CORRIDOR_PLAN = {
    "boundary": {
        "polygon": [[0, 0], [4, 0], [4, 8], [0, 8]],
        "entrance": [0, 1.0],
    },
    "rooms": [
        {"id": "r1", "type": "living_room",
         "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]},
        {"id": "r2", "type": "bedroom",
         "polygon": [[0, 4], [4, 4], [4, 8], [0, 8]]},
    ],
    "openings": [
        {"id": "d0", "type": "door", "host_wall": ["exterior", "r1"],
         "endpoints": [[0, 0.5], [0, 1.5]], "clear_width": 0.9},
        {"id": "d1", "type": "door", "host_wall": ["r1", "r2"],
         "endpoints": [[1.0, 4], [1.9, 4]], "clear_width": 0.9},
    ],
}

# Overlapping rooms — geometry_gate fails.
GEOM_INVALID_PLAN = {
    "boundary": {
        "polygon": [[0, 0], [9, 0], [9, 3], [0, 3]],
        "entrance": [0, 1.5],
    },
    "rooms": [
        {"id": "r1", "type": "bedroom",
         "polygon": [[0, 0], [5, 0], [5, 3], [0, 3]]},
        {"id": "r2", "type": "bedroom",
         # Overlaps r1 by a 1×3 m strip — triggers ROOM_OVERLAP in geometry_gate.
         "polygon": [[4, 0], [9, 0], [9, 3], [4, 3]]},
    ],
    "openings": [
        {"id": "d0", "type": "door", "host_wall": ["exterior", "r1"],
         "endpoints": [[0, 0.5], [0, 1.5]], "clear_width": 0.9},
    ],
}

# Valid geometry but r3 has no door connection — topology fails.
# Three adjacent rooms; d2 (r2 ↔ r3 link) is deliberately omitted.
TOPO_DISCONNECTED_PLAN = {
    "boundary": {
        "polygon": [[0, 0], [12, 0], [12, 2], [0, 2]],
        "entrance": [0, 1.0],
    },
    "rooms": [
        {"id": "r1", "type": "bedroom", "polygon": [[0, 0],  [4, 0],  [4, 2],  [0, 2]]},
        {"id": "r2", "type": "bedroom", "polygon": [[4, 0],  [8, 0],  [8, 2],  [4, 2]]},
        {"id": "r3", "type": "bedroom", "polygon": [[8, 0],  [12, 0], [12, 2], [8, 2]]},
    ],
    "openings": [
        {"id": "d0", "type": "door", "host_wall": ["exterior", "r1"],
         "endpoints": [[0, 0], [0, 2]], "clear_width": 0.9},
        {"id": "d1", "type": "door", "host_wall": ["r1", "r2"],
         "endpoints": [[4, 0], [4, 2]], "clear_width": 0.9},
        # d2 omitted — r3 is topologically isolated.
    ],
}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_fully_compliant_plan_scores_above_m():
    """All clauses pass → C_all=1, reward > M (= M + R_code + β*Q)."""
    result = compute_reward(COMPLIANT_PLAN)
    assert not result["hard_gate_fail"]
    assert result["gate_reason"] is None
    assert result["C_all"] == 1
    assert result["R_code"] == pytest.approx(1.0, abs=1e-6)
    # M + R_code + β*Q > M since R_code=1.0 and Q≥0.
    assert result["reward"] > _M
    # Every clause record should be PASS with non-negative c_k.
    for rec in result["clause_records"]:
        assert rec["c_k"] == 1, f"expected compliant record: {rec}"


def test_geometry_invalid_plan_returns_hard_gate():
    """Overlapping rooms → geometry_gate fails → R = −1."""
    result = compute_reward(GEOM_INVALID_PLAN)
    assert result["reward"] == pytest.approx(-1.0)
    assert result["hard_gate_fail"] is True
    assert result["gate_reason"] == "geometry"
    assert result["clause_records"] == []


def test_topology_disconnected_plan_returns_hard_gate():
    """Isolated room → topology fails → R = −1."""
    result = compute_reward(TOPO_DISCONNECTED_PLAN)
    assert result["reward"] == pytest.approx(-1.0)
    assert result["hard_gate_fail"] is True
    assert result["gate_reason"] == "topology"
    assert result["clause_records"] == []


def test_no_rooms_returns_hard_gate():
    """Empty room list → no_rooms guard → R = −1."""
    plan = {
        "boundary": {"polygon": [[0, 0], [10, 0], [10, 5], [0, 5]], "entrance": [0, 2.5]},
        "rooms": [],
        "openings": [],
    }
    result = compute_reward(plan)
    assert result["reward"] == pytest.approx(-1.0)
    assert result["gate_reason"] == "no_rooms"


def test_single_clause_violation_c_all_zero_reward_below_m():
    """Narrow corridor (0.80 m < 0.90 m) → CORR-W-01 fails → C_all=0, reward < M."""
    result = compute_reward(SINGLE_VIOLATION_PLAN)
    assert not result["hard_gate_fail"]
    assert result["C_all"] == 0
    assert result["reward"] < _M
    # R_code should still be positive (two clauses pass, one fails slightly)
    assert result["R_code"] > 0
    # Exactly one clause should have c_k=0 (the corridor)
    failing = [r for r in result["clause_records"] if r["c_k"] == 0]
    assert len(failing) >= 1
    assert all(r["clause_id"] == "CORR-W-01" for r in failing)


def test_multi_clause_violation_reward_lower_than_single_violation():
    """Two failing clauses → reward is strictly lower than single-violation plan."""
    single = compute_reward(SINGLE_VIOLATION_PLAN)
    multi = compute_reward(MULTI_VIOLATION_PLAN)
    assert multi["C_all"] == 0
    assert multi["reward"] < single["reward"]
    # Both CORR-W-01 and DOOR-W-01 must have at least one failing record
    multi_failing_clauses = {r["clause_id"] for r in multi["clause_records"] if r["c_k"] == 0}
    assert "CORR-W-01" in multi_failing_clauses
    assert "DOOR-W-01" in multi_failing_clauses


def test_borderline_plan_all_clauses_just_pass():
    """All measurements just above threshold → C_all=1, reward > M."""
    result = compute_reward(BORDERLINE_PLAN)
    assert result["C_all"] == 1
    assert result["reward"] > _M
    for rec in result["clause_records"]:
        assert rec["c_k"] == 1, f"expected borderline-pass record: {rec}"


def test_error_record_treated_as_max_severity_fail():
    """
    d2 missing clear_width → door.check returns ERROR → treated as c_k=0, s_k=1.0,
    effective_margin=−1.0 → DOOR-W-01 fails → C_all=0, reward < M.
    """
    result = compute_reward(MISSING_CLEARWIDTH_PLAN)
    assert not result["hard_gate_fail"]
    assert result["C_all"] == 0
    assert result["reward"] < _M

    error_recs = [r for r in result["clause_records"] if r["status"] == "ERROR"]
    assert len(error_recs) >= 1
    for r in error_recs:
        assert r["effective_margin"] == pytest.approx(-1.0)
        assert r["c_k"] == 0
        assert r["s_k"] == pytest.approx(1.0)


def test_no_corridor_rooms_vacuous_clause_pass():
    """
    Plan with no corridor-typed rooms → CORR-W-01 has no records.
    Vacuous pass: clause contributes its weight to R_code, C_all stays 1.
    """
    result = compute_reward(NO_CORRIDOR_PLAN)
    assert result["C_all"] == 1
    assert result["reward"] > _M
    assert result["R_code"] == pytest.approx(1.0, abs=1e-6)
    corr_records = [r for r in result["clause_records"] if r["clause_id"] == "CORR-W-01"]
    assert corr_records == []


def test_m_dominance_compliant_beats_all_violating(capsys):
    """
    M-dominance property: min(compliant rewards) > max(statutory-violating rewards),
    regardless of Q(y).

    Compliant set: COMPLIANT, BORDERLINE, NO_CORRIDOR
    Violating set: SINGLE_VIOLATION, MULTI_VIOLATION, MISSING_CLEARWIDTH

    Hard-gate plans (GEOM_INVALID, TOPO_DISCONNECTED) return −1 which is always below
    any violating plan's reward, so they are not included in the violating set here.

    The empirical gap (min_compliant − max_violating) is printed and should be recorded
    in FINDINGS.md after this test run.
    """
    compliant_plans = [COMPLIANT_PLAN, BORDERLINE_PLAN, NO_CORRIDOR_PLAN]
    violating_plans = [SINGLE_VIOLATION_PLAN, MULTI_VIOLATION_PLAN, MISSING_CLEARWIDTH_PLAN]

    compliant_rewards = [compute_reward(p)["reward"] for p in compliant_plans]
    violating_rewards = [compute_reward(p)["reward"] for p in violating_plans]

    min_compliant = min(compliant_rewards)
    max_violating = max(violating_rewards)
    gap = min_compliant - max_violating

    print(
        f"\nM-dominance check  (w=1/3, lam=0.5, M=2.0, beta=0.1)\n"
        f"  compliant rewards : {[f'{r:.4f}' for r in compliant_rewards]}\n"
        f"  violating rewards : {[f'{r:.4f}' for r in violating_rewards]}\n"
        f"  min_compliant     = {min_compliant:.4f}\n"
        f"  max_violating     = {max_violating:.4f}\n"
        f"  gap               = {gap:.4f}  (must be > 0)"
    )

    assert gap > 0, (
        f"M-dominance violated: min compliant reward {min_compliant:.4f} "
        f"<= max violating reward {max_violating:.4f}"
    )

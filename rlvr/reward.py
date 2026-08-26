"""
Reward function for verifier-grounded RLVR.

Formula (PROJECT_CONTEXT.md spec):

    R(y) = -1                                        if hard gate fails
    R(y) = M * C_all(y) + R_code(y) + β * Q(y)      otherwise

    C_all(y)  = 1 if all statutory clauses pass, 0 otherwise  (Π c_k)
    R_code(y) = Σ_k  w_k * c_k(y)  −  λ * Σ_k  w_k * s_k(y)
    c_k(y)    = 1[m_k(y) ≥ 0]        compliance indicator
    s_k(y)    = max(0, −m_k(y))       violation severity
    m_k(y)    = worst-case (minimum) normalised margin over all records for clause k

PLACEHOLDER training constants — tunable, not statutory thresholds (see DECISIONS.md 2026-08-26):
    w_k = 1/3  per clause  (equal weights, normalised to sum to 1 over 3 clauses)
    λ   = 0.5  severity penalty coefficient
    M   = 2.0  compliance domination bonus
    β   = 0.1  quality score coefficient

Hard gate — triggers R(y) = −1:
    1. geometry_gate fails  (structural invalidity makes statutory measurement meaningless)
    2. topology fails       (disconnected room would make egress.check return ERRORs)
    3. plan has no rooms    (vacuous compliance is not compliance; also catches schema gap
                             where geometry_gate silently passes on an empty rooms list)

    Note: PROJECT_CONTEXT.md refers to a "syntax/schema gate" as a precursor to
    geometry_gate, but no such validator exists yet in the codebase — see DECISIONS.md
    2026-08-26 for the gap flag and rationale for deferring a full schema gate.

ERROR records — treated as FAIL with synthetic margin = −1.0 (maximum severity):
    An ERROR record means the verifier could not produce a numeric measurement.
    The most common causes after the hard gate passes are:
      - door.check: missing or non-numeric clear_width field (schema gap)
      - egress.check: within-room waypoint graph is disconnected due to extreme concavity
    Treating ERROR as maximum-severity FAIL ensures the reward signal pushes away from
    plans that break any verifier, not just plans that produce numeric FAILs.
    The synthetic margin −1.0 is documented here (not just in DECISIONS.md) because it
    is a non-obvious design choice that affects how every ERROR record propagates into R_code.

Clause aggregation — worst-case margin per clause:
    Each statutory clause (CORR-W-01, DOOR-W-01, EGR-D-01) may produce multiple audit
    records (one per corridor room, one per door, one per room).  Building code requires
    every instance to comply, so the clause margin m_k is the minimum margin across all
    its records.  The severity s_k reflects the worst violation.

Vacuous clause pass:
    If a plan has no rooms of the relevant type (e.g., no corridor-typed rooms), the
    corresponding clause has no records and is treated as vacuously passing (c_k = 1,
    s_k = 0).  The clause weight still contributes to R_code.

Q(y) — PLACEHOLDER: area-weighted mean isoperimetric compactness (see DECISIONS.md 2026-08-26):
    compactness_k = 4π * area_k / perimeter_k²    (range (0, 1]; circle = 1, square ≈ 0.785)
    Q(y)          = Σ(area_k * compactness_k) / Σ(area_k)
    Area-weighted so large habitable rooms dominate; a small misshapen closet does not
    swamp the score of the living room.
"""

from __future__ import annotations

import math

from shapely.geometry import Polygon

from verifiers import corridor, door, egress, geometry_gate, topology

# PLACEHOLDER training constants — tunable, not statutory thresholds (see DECISIONS.md 2026-08-26)
_W: dict[str, float] = {
    "CORR-W-01": 1 / 3,
    "DOOR-W-01": 1 / 3,
    "EGR-D-01":  1 / 3,
}
_LAMBDA: float = 0.5    # severity penalty coefficient
_M: float = 2.0         # compliance domination bonus
_BETA: float = 0.1      # quality score coefficient
_HARD_GATE_REWARD: float = -1.0
# Synthetic margin assigned to ERROR records — maximum severity (see module docstring)
_ERROR_MARGIN: float = -1.0


def compute_reward(plan: dict) -> dict:
    """
    Compute R(y) for a single floor-plan JSON.

    Parameters
    ----------
    plan : dict
        Floor-plan JSON conforming to the project schema.

    Returns
    -------
    dict:
        reward         : float    — R(y)
        hard_gate_fail : bool
        gate_reason    : str | None   — "no_rooms", "geometry", "topology", or None
        C_all          : int      — 1 if all statutory clauses pass, 0 otherwise
        R_code         : float
        Q              : float
        clause_records : list[dict]  — annotated audit records (adds effective_margin,
                                       c_k, s_k to each raw verifier record)
    """
    # ── Minimal schema guard (no rooms = hard gate) ───────────────────────────
    if not plan.get("rooms"):
        return _hard_gate_result("no_rooms")

    # ── Structural gates ──────────────────────────────────────────────────────
    geo = geometry_gate.check(plan)
    if not geo["pass"]:
        return _hard_gate_result("geometry")

    topo = topology.check(plan)
    if not topo["pass"]:
        return _hard_gate_result("topology")

    # ── Collect statutory clause records ──────────────────────────────────────
    raw_records = corridor.check(plan) + door.check(plan) + egress.check(plan)

    # Annotate each record with effective_margin, c_k, s_k.
    # ERROR records receive the synthetic margin _ERROR_MARGIN (max severity).
    annotated: list[dict] = []
    for rec in raw_records:
        effective_m = _ERROR_MARGIN if rec["status"] == "ERROR" else rec["margin"]
        r = dict(rec)
        r["effective_margin"] = effective_m
        r["c_k"] = 1 if effective_m >= 0 else 0
        r["s_k"] = max(0.0, -effective_m)
        annotated.append(r)

    # ── Per-clause worst-case margin ──────────────────────────────────────────
    clause_worst: dict[str, float] = {}
    for rec in annotated:
        cid = rec["clause_id"]
        m = rec["effective_margin"]
        if cid not in clause_worst or m < clause_worst[cid]:
            clause_worst[cid] = m

    # ── R_code and C_all ──────────────────────────────────────────────────────
    R_code = 0.0
    C_all = 1
    for cid, w in _W.items():
        if cid not in clause_worst:
            # No records for this clause — vacuous pass (c_k=1, s_k=0).
            R_code += w
            continue
        m = clause_worst[cid]
        c_k = 1 if m >= 0 else 0
        s_k = max(0.0, -m)
        if c_k == 0:
            C_all = 0
        R_code += w * c_k - _LAMBDA * w * s_k

    # ── Quality score ─────────────────────────────────────────────────────────
    Q = _quality_score(plan)

    return {
        "reward": _M * C_all + R_code + _BETA * Q,
        "hard_gate_fail": False,
        "gate_reason": None,
        "C_all": C_all,
        "R_code": R_code,
        "Q": Q,
        "clause_records": annotated,
    }


def _hard_gate_result(reason: str) -> dict:
    return {
        "reward": _HARD_GATE_REWARD,
        "hard_gate_fail": True,
        "gate_reason": reason,
        "C_all": 0,
        "R_code": 0.0,
        "Q": 0.0,
        "clause_records": [],
    }


def _quality_score(plan: dict) -> float:
    """
    PLACEHOLDER: area-weighted mean isoperimetric compactness over room polygons.

    compactness_k = 4π * area_k / perimeter_k²    range (0, 1]; circle=1, square≈0.785
    Q             = Σ(area_k * compactness_k) / Σ(area_k)

    Area-weighted: a 16 m² living room influences Q more than a 2 m² closet.
    PLACEHOLDER — differentiating signal within the compliant region only; see DECISIONS.md.
    """
    total_area = 0.0
    weighted_sum = 0.0
    for room in plan.get("rooms", []):
        raw = room.get("polygon", [])
        if len(raw) < 3:
            continue
        poly = Polygon([tuple(p) for p in raw])
        if not poly.is_valid or poly.area < 1e-9:
            continue
        compactness = 4 * math.pi * poly.area / (poly.length ** 2)
        compactness = min(compactness, 1.0)   # clamp floating-point overshoot
        total_area += poly.area
        weighted_sum += poly.area * compactness
    if total_area < 1e-9:
        return 0.0
    return weighted_sum / total_area

# Findings

Dated log of what was tried, what worked, and what broke — with numbers where available.

---

## 2026-08-26 — Day 1: Verifier bank v1

### Verifiers written
- `verifiers/geometry_gate.py` — structural gate (polygon validity, mutual non-overlap, opening placement)
- `verifiers/corridor.py` — CORR-W-01, PLACEHOLDER threshold 0.90 m
- `verifiers/door.py` — DOOR-W-01, PLACEHOLDER threshold 0.80 m

### Test suite results
36 tests across 3 files — **36 passed, 0 failed** (pytest 9.1.1, Python 3.13.3, Shapely 2.1.2).

| File | Tests | Result |
|------|-------|--------|
| test_door.py | 13 | all pass |
| test_corridor.py | 10 | all pass |
| test_geometry_gate.py | 13 | all pass |

### L-shaped corridor verification
Manually traced the per-edge-normal algorithm on polygon
`(0,0)→(4,0)→(4,1)→(1,1)→(1,4)→(0,4)` (both arms 1.0 m wide):
- Edge `(4,1)→(1,1)`, normal `(0,−1)`: chords at x∈[1.3,3.7] → 1.0 m each ✓
- Edge `(1,1)→(1,4)`, normal `(−1,0)`: chords at y∈[1.3,3.7] → 1.0 m each ✓
- Other edges produce 4 m chords (arm lengths) — minimum wins.
Reported width: 1.0 m. Test `test_l_shaped_corridor_pass` confirms this (tolerance ±0.05 m).

The adversarial variant (both arms 0.8 m) also passes: `test_l_shaped_narrow_arm_fail`
confirms the algorithm reports 0.8 m and returns FAIL, ruling out the MBR-axis failure mode.

### Nothing broke
No near-boundary numerical failures observed. The 1e-9 m² overlap tolerance correctly
ignores shared-edge artefacts while catching a 0.001 m room encroachment (6e-3 m² overlap).

---

## 2026-08-26 — Day 2: Topology + Egress verifiers

### Verifiers written
- `verifiers/topology.py` — structural gate, BFS reachability from entrance node
- `verifiers/egress.py` — EGR-D-01, PLACEHOLDER threshold 30.0 m

### Test suite results
59 tests across 5 files — **59 passed, 0 failed** (pytest 9.1.1, Python 3.13.3, Shapely 2.1.2).

| File | Tests | Result |
|------|-------|--------|
| test_door.py | 13 | all pass |
| test_corridor.py | 10 | all pass |
| test_geometry_gate.py | 13 | all pass |
| test_topology.py | 11 | all pass |
| test_egress.py | 12 | all pass |

### Find-and-fix: margin formula normalization discrepancy
**Found**: corridor.py and door.py used unnormalized margin `(measured − threshold)`,
but PROJECT_CONTEXT.md reward spec requires normalized form
`(measured − threshold) / max(|threshold|, ε)`.

**Fix**: Added `_EPS = 1e-9` to both files and updated all three margin computations
(corridor.py ×1, door.py ×2) to use the normalized formula. Updated one explicit
margin value assertion in test_door.py: `pytest.approx(0.10)` → `pytest.approx(0.125)`
(was measuring unnormalized 0.90 − 0.80 = 0.10; correct normalized value is
(0.90 − 0.80) / 0.80 = 0.125).

Impact: margin values for corridor and door verifiers shifted (e.g., a 0.10 m excess
becomes a 0.111 normalized score, not 0.10). The pass/fail boundary is unaffected
(margin ≥ 0 ↔ measured ≥ threshold in both formulations).

### Concave room test: covers() filter confirmed effective
L-shaped room test (`test_concave_room_covers_filter`): without the filter the direct
diagonal from door midpoint (0, 5) to far corner (10, 0) would be ≈ 11.18 m (invalid —
exits the polygon). With the filter the measured value is > 13.5 m, confirming the
detour path through the L-joint is taken.

### Detour test: room-id node keys confirmed working
Room B accessible only via room A (not by cutting across the shared wall corner):
measured path > 14 m vs. straight-line ≈ 13.45 m. Confirms room-id node keys
prevent shared polygon corners from acting as inter-room bridges.

---

## 2026-08-26 — Reward function skeleton (Day 6-7 work, pulled forward)

### Module written
- `rlvr/reward.py` — `compute_reward(plan) -> dict`
  Hard gate (geometry_gate + topology + no_rooms) → R = −1.
  Statutory score R_code from corridor, door, egress margins.
  Quality score Q (area-weighted isoperimetric compactness, PLACEHOLDER).
  Full formula: R(y) = M * C_all + R_code + β * Q.

### Test suite results
10 tests in `rlvr/tests/test_reward.py` — **10 passed, 0 failed**.
Combined suite (verifiers + reward): **69 passed, 0 failed**
(pytest 9.1.1, Python 3.13.3, Shapely 2.1.2).

| File | Tests | Result |
|------|-------|--------|
| test_reward.py | 10 | all pass |
| (verifier tests unchanged) | 59 | all pass |

### M-dominance: empirical gap on test plans

Training constants: w_k=1/3, lam=0.5, M=2.0, beta=0.1

| Plan | Reward |
|------|--------|
| COMPLIANT (corridor 1.2 m, doors 0.9 m) | 3.0756 |
| BORDERLINE (corridor 0.91 m, doors 0.81 m) | 3.0754 |
| NO_CORRIDOR (no corridor rooms, doors 0.9 m) | 3.0785 |
| SINGLE_VIOLATION (corridor 0.80 m) | 0.7235 |
| MISSING_CLEARWIDTH (door ERROR record) | 0.5756 |
| MULTI_VIOLATION (corridor 0.80 m + doors 0.75 m) | 0.3798 |

**min compliant = 3.0754, max violating = 0.7235, gap = 2.3518**

The algebraic lower bound on the gap is ~2.233 (M + 1 − 2/3 − β).
The empirical gap (2.3518) exceeds this, confirming Q(y) does not cause any
violating plan to approach the compliant floor.  If training plans ever produce
a gap near zero, revisit M or β.

### Known edge cases not covered by tests
- A plan where egress returns ERROR after topology passes (extreme within-room
  concavity disconnecting the waypoint graph).  This case is handled by the
  ERROR-as-FAIL path in reward.py but is not yet exercised by a specific fixture.
  The MISSING_CLEARWIDTH test exercises the ERROR-as-FAIL path via door.check.
- Plans with no openings at all (would fail topology gate before reaching R_code).

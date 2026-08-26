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

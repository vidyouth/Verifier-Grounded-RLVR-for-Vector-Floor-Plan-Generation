# Findings

Dated log of what was tried, what worked, and what broke — with numbers where available.

---

## 2026-08-31 — Day 4-5: SFT data prep + scripts (training run not yet executed)

Approach and every hyperparameter choice: DECISIONS.md 2026-08-31. Fast-lane
work (data-prep + training-script plumbing); no verifier or `reward.py` touched.

### `sft/` delivered
- `plan_prompt.py` — single source of truth for prompt/completion + downstream
  reassembly (`reconstruct_plan`: templated metadata, `relations` derived from
  openings, `audit=[]`).
- `prepare_sft_data.py` — `data/processed/{split}.jsonl` → `sft/data/{split}.jsonl`
  instruction pairs. Needs only `shapely`; runs locally.
- `train_qlora.py` — plain one-command QLoRA script, wall-clock Drive
  checkpointing, auto-resume, one status line per log step.
- `eval_json_validity.py` — greedy decode over val, strict `json.loads` headline
  rate + `geometry_gate` secondary, per-plan resumable output.
- `requirements.txt` (pinned), `README.md` (verbatim Colab steps).

### Data prep run (local, char-based estimate — exact tokens computed in Colab)
`python sft/prepare_sft_data.py --splits train val`:

| split | pairs written | prompt chars p50/p90 | completion chars p50/p90/p99/max | est. tokens p50/p90/p99 |
|---|---|---|---|---|
| train | 10,788 | 1434 / 1644 | 5023 / 6701 / 8283 / 25986 | 2305 / 2965 / 3583 |
| val   | 1,349  | 1433 / 1634 | 5032 / 6557 / 8260 / 13874 | 2310 / 2902 / 3589 |

At `max_seq_len=3072` the train-time filter is expected to drop ~8-9% of train
pairs as over-length (exact % printed by `train_qlora.py` on the real tokenizer).
The one 26k-char outlier is a jagged-traced-contour plan (see Day 3 phase B
finding #2).

### Roundtrip check (local)
`build_completion_text` → `json.loads` → `reconstruct_plan` → `geometry_gate`:
**401/401** first val plans PASS. `relations.derived` rebuilt from openings
matches the converter's own list **401/401**. So the templated-downstream split
is faithful — the model is not being asked to produce anything the pipeline
can't deterministically restore.

### Regression
Full suite green: **110 passed** (`verifiers/tests/ rlvr/tests/ data/tests/`,
pytest, Python 3.13.3, Shapely 2.1.2). No verifier / reward file modified.

### Not done this session
The actual QLoRA training run and the JSON-validity eval — both need the T4.
Hyperparameters confirmed by user 2026-08-31. Next Colab session runs
`train_qlora.py --calibrate 50` first (measured s/step → real full-run estimate,
logged against the ~1 hr heuristic in DECISIONS.md) before the unattended run.

---

## 2026-08-30 — Day 3 phase A: ResPlan dataset inspection (read-only)

Switched data source RPLAN → ResPlan (see DECISIONS.md 2026-08-30). This session
was read-only inspection of the raw dataset; no conversion code written, no
verifier touched.

### Acquisition
- `git clone --depth 1 https://github.com/m-agour/ResPlan.git`, then
  `unzip ResPlan.zip` → `ResPlan.pkl` (247 MB, pickle of a `list[dict]`, 17,000
  entries). Files staged in `data/raw/`; `ResPlan.pkl` + `ResPlan.zip` git-ignored,
  everything small kept (see `data/raw/PROVENANCE.md`).
- Local env already had it covered: Python 3.13.3, Shapely 2.1.2. Geometry values
  in the pickle are **live Shapely objects**, not coordinate arrays.

### Plan schema (one real example: `data[0]`, `id=14433`)
Keys present: `id, area, net_area, wall_depth, inner, land, neighbor, living,
bedroom, bathroom, kitchen, balcony, storage, stair, door, window, front_door,
garden, parking, pool` (+ `area_change_sqft` on augmented plans only).

```
id          : 14433
area        : 120.7739                     # m^2, reliable
net_area    : 95.659...                     # UNRELIABLE (see below)
wall_depth  : 4.503                         # canvas units (~0.23 m)
inner       : MultiPolygon(n=1, area=46029.3 canvas^2)   # building interior
land        : MultiPolygon(n=1, area=60836.0)            # plot boundary
neighbor    : (MultiPolygon, MultiPolygon)  # near-degenerate, areas ~0.0-0.1 — ignore
living      : Polygon(area=18806.5, 17 verts)            # single polygon
bedroom     : MultiPolygon(n=3, areas=[5431, 6954, 4977])
bathroom    : MultiPolygon(n=3)
kitchen     : Polygon(area=3308.9, 9 verts)
balcony     : MultiPolygon(n=2)
door        : MultiPolygon(n=8, areas ~79-87)            # bare rectangles
window      : MultiPolygon(n=7)
front_door  : Polygon(4 verts, area 105)                 # exactly one per plan
garden/parking/pool : MultiPolygon(n=0)                  # empty here
```
Room polygons are axis-aligned rectilinear rings (living-room example: 17 vertices,
L-shaped). Coordinates are floats in a ~256-unit canvas (`inner` bounds
≈ (0, 19) → (256, 237)).

### Room-type vocabulary (measured over 300–17,000 plans)
Keys that ever hold room polygons: **living, bedroom, bathroom, kitchen, balcony,
storage, stair** (interior) and **garden, parking, pool** (outdoor). Frequencies
over first 300 plans: living 300/300, bedroom 300, bathroom 300, kitchen 296,
balcony 233, storage 28, stair 19.
- **Corridor / hallway / circulation: NO SUCH LABEL EXISTS.** Probed explicitly for
  `corridor, hallway, hall, passage, circulation, foyer, lobby, dining, study,
  laundry, utility, closet` — none present. The upstream README's "17-category
  taxonomy" counts sub-splits (e.g. `front_door`, `wall`, `window`, empty outdoor
  keys) that `resplan_utils.CATEGORY_COLORS` also enumerates; the functional
  interior room set is the 7 above. Circulation space in these plans is implicit
  (unlabeled leftover area between rooms), not a polygon.

### Doors / windows — how they reference rooms/walls  (our `host_wall` equivalent)
**They don't.** `door`, `window`, `front_door` are standalone MultiPolygon/Polygon
rectangles with no IDs and no room or wall references. Room-to-room connectivity is
purely geometric and must be *inferred*:
- Each `door` polygon is a thin rectangle embedded in a wall line — short side ≈
  `wall_depth` (~0.23 m), long side ≈ the clear opening (**median 0.95 m**, so
  DOOR-W-01 has a real measurable input). `window` long side median ~1.16 m,
  `front_door` ~1.19 m.
- `resplan_utils.plan_to_graph` recovers edges by buffering each opening by a
  scale-aware epsilon and testing which room polygons it intersects; a door
  bridging exactly 2 rooms → that edge. Our conversion must do the same to fill
  `host_wall: [room_a, room_b]` (and `["exterior", room_x]` for `front_door`).
- `wall` is a MultiPolygon of ~9 thick chunks per plan, not a connected network.

### Connectivity graph edge types (`via_door` / `adjacency` / `via_window` / `direct`)
The **repo pickle carries geometry only** — `plan["graph"]` is not populated
(Kaggle release has it prebuilt). Edge semantics, from `resplan_utils.py`:
- `via_door` / `via_window` — two room polygons both intersect the buffered
  door/window connector geometry (a real walkable opening / a window between them).
- `adjacency` (paper/Kaggle) = strict `via_opening` (a contiguous wall-free segment
  ≥ ~2×wall_depth along the shared boundary — open kitchen, archway) **plus**, via
  `add_adjacency_edges`, any two rooms within one wall thickness of each other
  (i.e. share a wall, no opening). So `adjacency` conflates "walk-through opening"
  and "just share a wall".
- `direct` — `front_door` node ↔ each room its polygon physically touches.
- `fallback` — nearest-neighbour hack to keep the graph one connected component;
  not a real physical relation.
Strict `plan_to_graph` ≈ 8.7 edges/plan over 5 types; paper definition ≈ 12.9 over
4 types. For our egress/topology needs, only `via_door` + `front_door`→`direct`
represent actually-walkable passages.

### Room-count distribution (all 17,000 plans; our locked scope is 5–8)
Count = living + bedroom + bathroom + kitchen + balcony + storage + stair:
```
 4:   18    5:  697    6: 2398    7: 4252    8: 3379
 9: 2472   10: 1689   11:  849   12+: ~1250
 --> 5–8 inclusive: 10,726 / 17,000 = 63.1%
```
Excluding balcony/storage/stair (living+bedroom+bathroom+kitchen only):
5–8 = 15,528 / 17,000 = **91.3%**. So the scope yield swings from ~63% to ~91%
depending on whether balconies count as "rooms" — this needs a phase-B decision.
Bedrooms/plan: 1→1796, 2→8007, 3→6109, 4→819, 5+→266. Bathrooms and balconies are
frequently multiple (drives the high total counts).

### split.json granularity
Flat, **plan-level**. Keys: `train` 13053, `val` 1632, `test` 1632, `augmented`
683 — 17,000 unique integer IDs, zero overlap between any two lists, and every ID
equals a `plan["id"]` in the pickle (1:1, no missing). **No building / listing /
parent grouping field exists** anywhere in the schema, so "split by parent
building" (EXECUTION_PLAN) cannot lean on a dataset-provided identity — we either
trust ResPlan's canonical split or re-dedup geometrically ourselves.

### Augmented plans
683 plans (`split.json["augmented"]`, IDs 100708–999398) are geometric
augmentations (rotate/flip/scale) of 667 originals. They ARE present in the pickle,
mixed into the 17,000. Two independent ways to flag them: (a) membership in the
`augmented` list; (b) presence of the `area_change_sqft` key — these two sets
match exactly (683 = 683). They never appear in train/val/test lists. **No explicit
field links an augmented plan to its origin** — an exact rounded-coordinate hash of
the `inner` polygon matches 667 augmented↔origin pairs (flip/scale preserves the
sorted rounded-vertex set), so origin recovery is geometric if ever needed.

### Duplicates / near-duplicates in-sample
Cheap check (hash of `inner` exterior vertices rounded to 0.1) over 17,000 plans:
757 exact-match clusters / 1,541 plans — but 667 of those clusters are the
augmented↔origin pairs above. Removing those leaves **90 clusters (~180 plans) of
genuinely redundant original listings** (e.g. IDs [8266, 5647], [14129, 14128],
[29, 30]). Upstream's fuller geometry-based scan (not just exact hash) reports 931
clusters / 1,170 plans (6.9%). Coarse (area, bbox, room-count, bedroom-count)
bucketing flags a similar order of near-dupes.

### Other schema notes
- `area` (m²) is reliable. **`net_area` is not** — values seen: 95.7, 207.8, 5.97,
  0 on consecutive plans; does not track `area`. Do not use it.
- Coordinate unit ≠ metres (upstream README is wrong on this). Per-plan linear
  scale = `sqrt(inner_area_canvas / area_m2)` ≈ 18.3 canvas units/m (median;
  p10–p90 of the area ratio 155–559, i.e. scale varies per plan). `wall_depth`
  median 4.17 units ≈ 0.23 m, consistent with the README's "10–40 cm" claim once
  converted.
- `inner` is always a MultiPolygon: ~98% single-part, ~2% multi-part, ~1.7% have
  interior holes. `land` ⊃ `inner`. `neighbor` is a 2-tuple of near-degenerate
  MultiPolygons spanning the inner bbox — not room connectivity; ignore.
- 683/17,000 plans carry `area_change_sqft` (augmented only, per above).

### Nothing broke
Read-only session. No code written against the dataset, no verifier or reward file
touched. `data/raw/` staged, `.gitignore` updated, `data/raw/PROVENANCE.md` added.
Phase B (conversion mapping) picks up from these facts.

---

## 2026-08-30 — Day 3 phase B: conversion pipeline results

Pipeline: `data/convert.py` + `data/dedup_split.py` + `data/run.py`, 39 pipeline
tests (`data/tests/`) + the 69 verifier/reward tests all pass. Approach in
DECISIONS.md 2026-08-30. No verifier file modified.

### Plans surviving each stage (full 17,000-plan run)

| Stage | Plans | Δ |
|---|---|---|
| Raw ResPlan | 17,000 | — |
| After `has_stair` drop | 16,315 | −685 |
| After room-count ∈ [5,8] (living+bedroom+bathroom+kitchen+storage) | **14,357** | −1,958 |
| After `room_overlap_large` (raw pair > 0.5 m²) | 13,604 | −753 |
| After `room_too_small` / `invalid_room_geometry` / `room_overlap_residual` / `no_entrance` | 13,485 | −119 |
| **Converted, `geometry_gate` PASS** | **13,485 (79.3%)** | — |

`geometry_gate` pass rate on converted output: **13,485 / 13,485 = 100%** (by
construction — plans that don't clear it are dropped and counted above; final run
produced zero such drops).

### Verifier-bank behaviour on converted plans (3,000-plan random sample of train)

| Verifier | Result |
|---|---|
| `geometry_gate` | 3,000 / 3,000 PASS (100%) |
| `topology` | 2,400 / 3,000 PASS (**80.0%**); of the 600 fails, 557 have exactly 1 unreachable room |
| `corridor` (CORR-W-01) | **0 records on every plan** — vacuous pass, as designed (no corridor rooms exist) |
| `door` (DOOR-W-01) | 23,838 PASS / 1,513 FAIL records → **94.0%** of doors ≥ 0.80 m placeholder threshold |
| `egress` (EGR-D-01) | 19,867 PASS / 3,346 ERROR / 5 FAIL records → **85.6% PASS, 14.3% ERROR** |
| `topology` PASS **and** all `egress` PASS | 650 / 3,000 = **21.7%** |
| `compute_reward` hard-gate rate | **20.0%** (all from `topology`; geometry never fails, no empty-rooms) |

So of converted plans: ~22% are fully clause-compliant (reward ≥ 3), ~58%
convert but fail a clause (dominated by egress ERROR), ~20% hard-gate.

### Corridor synthesis: negative result (worth recording)

The residual-interior-area method (`inner − ∪ rooms`, rooms inward-buffered) was
implemented and measured over 250 in-scope plans at four inward-buffer settings.
On ~100% of plans it yields **the inter-room wall-gap web, not corridor shapes**:
surviving components have fill ratio (area ÷ bbox area) ≈ 0.06–0.14 and bounding
boxes spanning the whole floor plate, because ResPlan rooms tile the plate
wall-to-wall with only ~0.16 m structural gaps between them. The only compact
high-fill residuals were (a) sub-1 m² closet nooks and (b) 30–60 m² blobs on the
jagged-traced-contour plans. **There is no architect-authored circulation space
in ResPlan to recover** — circulation runs through rooms via doors. Corridor
synthesis abandoned; CORR-W-01 documented as a vacuous pass for this dataset
(LIMITATIONS.md).

### Dedup + split

- 12,757 clusters from 13,485 plans; **697 multi-plan clusters** (augmented↔origin
  pairs + republished near-duplicate listings); largest cluster 4 plans.
- Cluster-level 80/10/10 by plan count: **train 10,788 / val 1,349 / test 1,348**.
  No cluster spans two splits. ResPlan's canonical `split.json` was not used.

### Verifier issues surfaced by real converted data (reported, NOT patched)

1. **`egress` within-room concavity limitation is hit often on real geometry.**
   14.3% of egress records ERROR ("vertex not reachable from entrance") on
   *topology-passing* plans — mostly kitchens (582 of ~800 ERROR rooms in an
   earlier full-train pass) and other concave/complex rooms. egress v1 models
   within-room travel as straight segments from a single door midpoint to every
   vertex (`covers()` filter); real floor-plan rooms are frequently concave
   enough to disconnect some vertices. This is the documented v1 limitation
   (egress.py docstring, LIMITATIONS.md) now quantified — not a conversion defect.
   Candidate v2 fix: intermediate waypoints at concavity junctions / a proper
   visibility graph.
2. **Jagged traced contours reach ~435 vertices** on the worst converted room
   (`data/samples/sample_jagged_polygon_id4296.json`). `geometry_gate` and
   `corridor` handle it, but `egress`'s O(V²) `covers()` loop makes the full
   13,485-plan verifier sweep take >20 min — a handful of monster polygons
   dominate. Consider a vertex-count cap or polygon simplification before egress,
   or dropping plans with any room > ~120 vertices at conversion time
   (0.02% of ResPlan rooms per upstream).
3. **Topology "via_door + direct only" excludes open-plan connections.** ~20% of
   converted plans have an open-plan kitchen (or similar) that ResPlan encodes as
   `adjacency` with no connector polygon; by the confirmed edge-rule these are
   not walkable, so the plan fails `topology` and hard-gates. This is a
   consequence of the locked decision, not a bug — flagged in case the rule
   should be relaxed for wide unwalled openings (would recover most of the 20%).

### Nothing broke in the verifier bank
All 69 verifier/reward tests still green; no verifier or `reward.py` file touched.
Bulk splits + manifest in `data/processed/` (git-ignored); 9 converted samples in
`data/samples/` (committed, all pass `geometry_gate`).

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

# Converted sample plans

Nine ResPlan plans run through `python -m data.run` and written in the project
floor-plan JSON schema. Committed for reference and used by
`data/tests/test_convert.py::test_committed_sample_passes_geometry_gate`.

Regenerate with `python -m data.run --emit-samples` (selection is deterministic
given the same `ResPlan.pkl`). Bulk converted splits live in `data/processed/`
(git-ignored).

| File | Source `plan_id` | Illustrates |
|---|---|---|
| `sample_clean_exemplar_id17001.json` | 17001 | Fully clean: geometry_gate + topology pass, every egress record PASS |
| `sample_clean_exemplar_id17050.json` | 17050 | Fully clean |
| `sample_clean_exemplar_id2818.json` | 2818 | Fully clean |
| `sample_roomcount_5_min_id5616.json` | 5616 | Lower scope bound — 5 counted rooms (living+bedroom+bathroom+kitchen+storage) |
| `sample_roomcount_8_max_id14433.json` | 14433 | Upper scope bound — 8 counted rooms |
| `sample_open_plan_promoted_window_id14926.json` | 14926 | An open-plan opening ResPlan traced as a `window`, promoted to a walkable `door` (`id` still `window_*`) |
| `sample_exterior_window_id2372.json` | 2372 | Exterior windows kept as `type: "window"`, `host_wall = ["exterior", room]` |
| `sample_overlap_fixed_id7067.json` | 7067 | Raw plan had a room-pair tracing overlap; the smaller room yielded the overlap region during conversion |
| `sample_jagged_polygon_id4296.json` | 4296 | Worst-case traced contour in the dataset (~435-vertex room); still passes geometry_gate |

All nine pass `geometry_gate` and `topology`. Some carry `egress` `ERROR` /
`FAIL` records — that is the egress v1 within-room-concavity limitation surfacing
on real floor-plan geometry, documented in `docs/FINDINGS.md` (2026-08-30) and
`docs/LIMITATIONS.md`, not a conversion defect.

### Schema (per plan)

```
metadata : schema_version, jurisdiction (placeholder), source {dataset, plan_id, license}
           scale_canvas_units_per_m, room_count_in_scope
boundary : {polygon: [[x,y]...] (metres), entrance: [x,y]}
rooms    : [{id, type, polygon}]              type in living|bedroom|bathroom|kitchen|storage|balcony
openings : [{id, type, host_wall, endpoints, clear_width, swing}]
           type in door|window ; host_wall = [room_a, room_b] or ["exterior", room]
relations: {derived: [{type: via_door|direct, rooms: [...]}]}   # via_door + front-door only
audit    : []                                                    # inert placeholder
```

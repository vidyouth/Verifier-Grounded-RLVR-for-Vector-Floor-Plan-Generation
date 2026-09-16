# Limitations

Clause thresholds are placeholder/generic code, not a real statutory jurisdiction — results are not evidence of real-world compliance.

## Data source differs from the paper this project extends (added 2026-08-30)

This project was planned to use RPLAN / the Lara et al. dataset. Because the RPLAN
access request did not resolve within the fixed solo timeline, the data source was
switched to **ResPlan** (github.com/m-agour/ResPlan, CC BY 4.0). See
`docs/DECISIONS.md` 2026-08-30.

Implications:
- Any claim of consistency with, or reproduction of, the Lara et al. paper's
  results is limited by this substitution. Numbers here are computed on a
  different dataset with a different geometry pipeline and cannot be read as a
  like-for-like comparison.
- ResPlan's own stated scope limits carry over: all plans are from South Asian
  residential markets (layout conventions not globally representative); single
  floor, no furniture/3D; wall thickness is normalized per plan (10–40 cm for
  99.3% of plans), so within-plan wall-type variation is lost; a small tail of
  room polygons has jagged traced contours (0.52% > 30 vertices); ~6.9% of plans
  are near-duplicate republished listings, and 154 of the 1,632 canonical test
  plans have a near-duplicate in the canonical training split.
- ResPlan has **no corridor / hallway / circulation room type**. The CORR-W-01
  clause (minimum corridor clear width) has no applicable geometry on converted
  plans and is currently a documented no-op in the reward (vacuous clause pass).
  This weakens any "3-clause" framing until resolved (Day 3 phase B).

CC BY 4.0 attribution for ResPlan is **now present** in the top-level `README.md`
("Data" section); placement rationale in `data/raw/PROVENANCE.md`.

## Conversion-pipeline limitations (added 2026-08-30, Day 3 phase B)

The ResPlan → project-schema pipeline (`data/`) introduces its own limits.

### Corridor regions are geometrically inferred — and on this dataset, absent
No converted plan contains a room typed `corridor`. Corridor synthesis (residual
interior area → connected components → corridor polygons) was implemented and
**abandoned**: on ~100% of ResPlan plans the residual is the inter-room wall-gap
web, not corridor-shaped space, because ResPlan rooms tile the floor plate
wall-to-wall. `corridor.check` therefore returns `[]` for every converted plan
and **CORR-W-01 is a vacuous pass** (`c_k=1, s_k=0`) throughout training on this
data. Any "minimum corridor clear width" result on the ResPlan-derived set is
vacuous, not evidence of anything. Had corridors been synthesised, they would
still be *inferred geometry, not architect-authored* — a weaker basis than a
labelled room. See FINDINGS.md 2026-08-30.

### Canonical ResPlan split was NOT used
Train/val/test are produced by our own geometric dedup + cluster-level split
(IoU ≥ 0.90 near-duplicate clustering, augmented↔origin linked by inner-polygon
hash, whole clusters assigned 80/10/10 by plan count). ResPlan's `split.json`
is not used, because it has known train/test leakage (154 test plans with a
near-duplicate in train) and no building-identity field to group by. Our split
removes exact/near-duplicate leakage across the boundary but is not directly
comparable to any split used in the ResPlan paper or its baselines.

### Openings, host walls, and circulation are reconstructed, not read
ResPlan openings carry no room/wall references. `host_wall` is inferred by
buffer-and-intersect, and adjacent rooms (drawn ~0.16 m apart) are pulled
together with a local synthetic "gap-close" patch at each opening so the schema's
endpoint-on-wall invariant holds. Interior openings ResPlan traced as *windows*
are emitted as walkable *doors* (matching upstream's own heuristic). Downstream
verifier measurements are only as good as these reconstructions.

### Topology / egress consequences on the converted set (measured)
- **~20% of converted plans fail `topology`** (hard-gate in the reward). Almost
  always a single unreachable room — an open-plan kitchen connected in ResPlan
  only by an unwalled opening with no connector polygon (`adjacency`), which the
  locked "via_door + direct only" edge rule does not treat as walkable.
- **~14% of `egress` records ERROR** on topology-passing plans (mostly concave
  kitchens) — the within-room straight-segment limitation below, hit far more
  often on real ResPlan geometry than on the synthetic verifier tests.
- Net: only ~22% of converted plans are fully clause-compliant. This is expected
  for RLVR *seed* data (SFT learns structure; GRPO optimises compliance), but the
  low starting compliance rate and its causes should be stated in any write-up.

### Jagged traced contours
The worst converted room has ~435 vertices (`data/samples/sample_jagged_polygon_
id4296.json`). Verifiers produce correct results but `egress` is O(V²); a
vertex-count cap or simplification before egress is advisable at scale.

## SFT limitations (added 2026-08-31, Day 4-5)

- **Base model downgraded to 1.5B.** Per PROJECT_CONTEXT.md §3 the T4 free tier
  cannot train a 7-8B QLoRA in scope, so the policy is Qwen2.5-1.5B-Instruct
  (logged decision, DECISIONS.md 2026-08-31). Any generation-quality ceiling
  from model size applies to every downstream result (SFT and GRPO).
- **`max_seq_len = 3072` drops the large majority of plans, not a long tail.**
  Original estimate (2026-08-31, chars/2.8 heuristic): ~8-9%. **Measured on the
  real Qwen2.5-1.5B-Instruct tokenizer (2026-09-05): 88.55% of train / 89.03%
  of val pairs exceed the budget and are dropped** (not truncated) — the
  heuristic badly underestimated tokens-per-char for coordinate-heavy JSON.
  Effective train set at this cap is ~1,235 of 10,788 plans, skewed toward the
  smallest layouts in the 5-8-room scope. This is an open decision
  (DECISIONS.md 2026-09-05), not yet resolved — SFT has not been run on this
  filtered set at scale.
- **`metadata`, `relations`, `audit` are not model outputs.** They are templated
  or derived from `openings` after generation. The SFT model is never evaluated
  on its ability to produce them, and any downstream "schema completeness" claim
  must note that these keys are supplied by the pipeline, not the model.
- **JSON-validity rate is for greedy decoding.** The Day 4-5 headline metric uses
  deterministic `do_sample=False` decoding. Sampling (as used in GRPO rollouts)
  will produce a different, generally lower, raw-validity rate.
- **Coordinates carried at 6-dp metres.** Completions reproduce the converter's
  full-precision coordinates; this is token-expensive and inflates sequence
  length (hence the drop rate above). Not reduced because coarser rounding
  breaks `geometry_gate`'s endpoint-on-wall tolerance in the training targets.
- **CORR-W-01 remains vacuous** on this data (inherited from the ResPlan
  conversion — see above). SFT does not change that.
- **Training compute is hardware-dependent and not yet validated on the T4
  this plan assumed.** Measured locally (2026-09-05) on an RTX 4050 Laptop
  (6 GB VRAM): 123.374 s/step, projecting ~8:25–8:58 for a 3-epoch run — ~8-9×
  the ~1 hr T4 heuristic on record from 2026-08-31. No OOM at this GPU's 6 GB,
  but headroom was thin (~500-600 MB free at peak). Any runtime or feasibility
  claim in this project should specify which hardware it was measured on.

## SFT completion truncation — forward-flag for Day 6-7 GRPO (added 2026-09-12)

The Day 4-5 JSON-validity eval (300-plan random val sample, FINDINGS.md
2026-09-11/12) found that **99.4% of invalid completions (158/159) are
truncated, not malformed** — the model hits the `max_new_tokens=5632` budget
before emitting EOS, and this is sharply room-count-correlated (100% of
5/7/8-room invalid completions are truncation; 8-room plans have the lowest
overall valid_json rate at 34.0%). Only 1/300 completions were genuinely
malformed JSON.

**GRPO rollouts will hit this same wall**, especially on 7-8-room plans, since
rollout generation uses the same base model/tokenizer and produces the same
coordinate-heavy JSON. Two consequences for the reward function
(`rlvr/reward.py`) and rollout config, not yet implemented:
- **Rollout `max_new_tokens` must be sufficiently generous** — reusing 5632
  as-is would reproduce this truncation rate inside training, not just at
  eval time.
- **The reward function must treat truncation as distinct from a genuine
  clause violation**, not fold both into the same penalty. A truncated
  completion is missing information (incomplete JSON), not a plan that was
  fully generated and then measured as non-compliant — conflating the two
  would teach the policy the wrong thing (either overpenalizing near-complete
  long plans, or making truncation and real geometric failure
  indistinguishable in the training signal).

## Within-room convexity approximation (egress.py)

The egress verifier models within-room travel as straight-line segments between
polygon vertices and door midpoints. Each candidate segment is accepted only when
`Shapely.covers(LineString)` confirms the segment lies entirely inside the room
polygon. This is exact for convex rooms.

For concave rooms (L-shaped, notched, or otherwise non-convex):
- The covers() filter correctly blocks segments that would cut through a notch.
- However, some vertex-to-door-midpoint paths may be blocked even though a valid
  in-room path exists — the path would need to bend around the concavity, but the
  waypoint graph can only travel in straight lines between its fixed waypoints.
- In practice this means the measured egress distance for a heavily concave room
  may be reported as an ERROR ("vertex not reachable") even when it is physically
  reachable. The workaround is to add intermediate waypoints at concavity junctions,
  which is not implemented in v1.

The `test_concave_room_covers_filter` test exercises the L-shape case and confirms
the detour path is taken (measured > 13.5 m vs. straight-line ≈ 11.18 m). Rooms
with severe concavity beyond this L-shape approximation should be validated manually.

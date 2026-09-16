# Findings

Dated log of what was tried, what worked, and what broke — with numbers where available.

---

## 2026-09-12 — Pre-Day 6-7 diagnostic: reward-function variance on real SFT outputs — FLAT -1, zero variance

**Question**: before starting GRPO, does `rlvr/reward.py` (Day 3) produce any usable
spread across real model outputs at the current SFT checkpoint, or is it a flat -1
spike? This determines whether GRPO's group-relative advantage has any signal to
work with at all.

**Method**: patched `sft/eval_json_validity.py` with an opt-in `--save-raw-text` flag
(no change to default behaviour) and generated a fresh, naturally-distributed 40-plan
random sample of val (seed 202; room-count mix 20.0/45.0/17.5/17.5% for 5/6/7/8 rooms,
close to the true val mix) against `checkpoints/sft_qlora_5632/final`,
`max_new_tokens=5632`, greedy — same settings as the 300-plan eval. This is new
generation (40 plans, ~62 min at ~92s/plan), done specifically because the earlier
300-plan eval never persisted raw completion text, only pass/fail diagnostics — there
was no way to run `compute_reward` on already-generated outputs without it. For each
of the 40 raw completions: parsed JSON (fence-stripped fallback, matching
`eval_json_validity.py`'s own logic), reconstructed the full plan via
`sft/plan_prompt.reconstruct_plan`, and ran `rlvr/reward.py::compute_reward`
unmodified. Read-only diagnostic — `reward.py` itself was not touched, no Day 6-7
work started. Script: not committed (throwaway diagnostic), output saved to
`checkpoints/sft_qlora_5632/reward_variance_sample40.json`.

One explicit assumption, since `compute_reward` requires a dict and has no path for
"didn't parse": unparseable completions (25/40 truncated, plus any genuinely malformed)
were scored as the hard-gate reward (-1.0), on the reasoning that an unparseable output
is at least as bad as a structurally-invalid one and the project's own gap flag
(`rlvr/reward.py` docstring lines 27-29: "no such [syntax/schema] validator exists yet")
already anticipates this case belonging at the hard-gate floor. This is **not** something
`compute_reward` does natively — it is an assumption applied on top, disclosed here so
it can be revisited when the schema gate is actually built.

### Result: all 40/40 samples scored exactly R = -1.0000. Zero variance.

| stat | value |
|---|---|
| min | -1.0000 |
| max | -1.0000 |
| mean | -1.0000 |
| stdev | 0.0000 |

Histogram: 40/40 in the "== -1 (exact hard gate)" bucket; every other bucket (including
all of `(-1, 0)`, the soft-score-but-below-M-dominance range) is empty.

By category — **no category differs from any other, all are exactly -1**:

| category | n | min | max | mean | stdev |
|---|---|---|---|---|---|
| truncated (unparseable, hit token budget) | 25 | -1.0 | -1.0 | -1.0 | 0.0 |
| geometry-invalid (valid JSON, fails `geometry_gate` or raises `GEOSException` reconstructing) | 15 | -1.0 | -1.0 | -1.0 | 0.0 |
| clause-fail (passes geometry+topology, ≥1 statutory clause fails) | 0 | — | — | — | — |
| fully-passing | 0 | — | — | — | — |

**Zero samples reached the clause-fail or fully-passing categories** — every single
generation in this 40-plan sample either failed to parse/reconstruct or failed
`geometry_gate` outright (consistent with the 300-plan eval's 0/141
`geometry_gate_pass` rate, FINDINGS.md 2026-09-11/12). This means `R_code` and `Q`
(the soft statutory-margin and quality terms) were **never exercised on a single real
model output** in this sample — every plan hit the hard gate before those terms could
run.

### Answering the diagnostic question directly

**There is NOT meaningful variance. This is a flat -1 spike, not a distribution with
spread.** Both failure categories tested (truncated, geometry-invalid) are
indistinguishable from each other and from the hard-gate floor — not "somewhat worse
truncation vs. somewhat-less-bad geometry failure," but bit-for-bit identical reward.

**This is a real concern for Day 6-7, flagged plainly, not glossed over**: if GRPO
rollout groups at the current SFT checkpoint are sampled the way this eval sampled
(greedy-equivalent quality), every member of a group would likely score -1, making the
group-relative advantage ≈ 0 for every group — no learning signal, regardless of how
the advantage is computed. Two things this does NOT prove, and should not be
overstated as proving:
- GRPO rollouts use **sampling** (temperature/top-p), not greedy decoding — this
  40-plan sample and the 300-plan eval are both greedy, so real rollout diversity may
  produce some geometry-valid completions the greedy-decode ceiling does not reach.
  This is the single biggest reason not to treat this result as fully decisive.
- n=40 (and n=300 before it) is small; a soft-score floor this flat is a strong signal
  but not proof that literally zero geometry-valid completions are reachable at this
  checkpoint under any sampling.

**Practical implication for Day 6-7 design (not yet acted on — no reward.py or GRPO
code changed)**: if early GRPO rollouts show the same flat -1 pattern under real
sampling, the reward function alone will not rescue training — the SFT checkpoint's
geometry-validity rate needs to be nonzero for *some* fraction of a rollout group
before group-relative advantage has anything to differentiate. This should be checked
empirically with actual GRPO rollout sampling before assuming the reward function
design is sufficient as-is.

---

## 2026-09-11/12 — Day 4-5 closing: JSON-validity eval, ~300-plan random sample

Ran `sft/eval_json_validity.py` against `checkpoints/sft_qlora_5632/final`,
`--max-new-tokens 5632`, greedy decoding, on a **random sample of 300 of the
1,349 val plans** (`random.seed(42)`, unweighted `random.sample` — not the
7-8-room-weighted calibration probe used earlier). Sample room-count mix
(11.3/52.0/20.0/16.7% for 5/6/7/8 rooms) closely tracks the true val
distribution (11.5/50.5/19.3/18.8%), confirming the sample is representative
rather than skewed. Runtime ≈ 7.5 h at ~90 s/plan (batch size 8) — **this is a
sampling-based result, not a full-val-set eval**; the full 1,349-plan set was
explicitly not run, per the time/cost tradeoff.

### Headline: JSON-validity rate = 47.0% (141/300)

This is one finding straddling two independent, differently-caused failure
modes — reported separately, not conflated:

**(a) Truncation is the dominant failure mode, and it is room-count-correlated.**
Of the 159 invalid completions, **158/159 (99.4%) hit the `max_new_tokens`
budget** (no EOS token emitted) rather than being malformed JSON. Only
**1/300 (0.3%) was genuinely malformed** (unparseable JSON with an EOS token
already emitted, i.e. the model stopped generating valid-looking-but-broken
output) — consistent with the earlier calibration probe's 0% genuinely-
malformed finding, now measured on the representative sample as a small but
nonzero rate. By room count, truncation is worst at the complex end of scope:

| rooms | n | valid_json rate | invalid | of which truncated |
|---|---|---|---|---|
| 5 | 34 | 55.9% | 15 | 15/15 (100%) |
| 6 | 156 | 47.4% | 82 | 81/82 (98.8%) |
| 7 | 60 | 51.7% | 29 | 29/29 (100%) |
| 8 | 50 | **34.0%** | 33 | 33/33 (100%) |

This is a token-budget / training-exposure limitation (5632 was chosen as the
`max_seq_len` that retained 95.05%/95.8%/81.4% of train/7-room/8-room plans —
DECISIONS.md 2026-09-06 — so the model saw the fewest complete 8-room examples
during SFT), **not a format-learning failure**: 0% genuinely malformed across
every sample tested (calibration probe + this 300-plan sample combined) means
the model has reliably learned JSON syntax; it simply isn't finishing the
longest completions within budget.

**(b) `geometry_gate_pass` rate among valid, non-truncated outputs is 0%
(0/141).** Of the 141 valid-JSON completions, 94 raised a `GEOSException`
during `geometry_gate` reconstruction (self-intersecting/degenerate polygons)
and the remaining 47 reconstructed cleanly but still failed the gate outright
— none passed. **This is expected pre-RL**: SFT trains surface format
(syntactically valid JSON matching the schema), not geometric compliance: no
term in the SFT loss rewards polygon validity, wall-thickness consistency, or
opening placement. This 0% is exactly the gap GRPO's hard feasibility gate
(`rlvr/reward.py`, R = −1 below the gate) is designed to close — it is the
baseline the reward function needs to move off of, not a regression to
investigate.

### Not done
Full 1,349-plan val set was not run (time/cost tradeoff, per instructions).
This sample-based rate should not be read as more precise than its n=300
(95% CI roughly ±5.7pp on the headline rate under a binomial approximation).

---

## 2026-09-11 — Day 4-5: SFT training run completed successfully

Full 3-epoch QLoRA run at `max_seq_len=5632` (final config, DECISIONS.md
2026-09-06) ran continuously to completion. Closes out the SFT training
portion of Day 4-5.

### Result
- **Completed cleanly**: 1920/1920 optimiser steps, `epoch=2.9959` (~3.0), no
  crash, no `Traceback` anywhere in the full log. Process exited on its own
  after the final save; GPU confirmed idle afterward (643 MiB / 1% util, no
  lingering `train_qlora` process).
- **Total wall-clock: 428,722.7 s = 7145.4 min ≈ 119.1 hours ≈ 4 days 23.1
  hours.** Started 2026-09-06 10:50 UTC, completed 2026-09-11 09:56:41 UTC.
  This is well above the original 5632 calibration projection (~5.76 days /
  138.3 h) but in the same ballpark — measured pace ran a bit faster than
  calibration for most of the run (~210-224 s/step vs. the 258.84 s/step
  calibration figure), landing close to the projection overall.
- **Final train loss: 0.1993** (last logged step); **mean train loss over the
  whole run: 0.2331**.
- **Final eval_loss: 0.2035** (step 1800, epoch 2.81 — the last scheduled
  eval before completion at step 1920). Full eval trend across all 9
  scheduled evals, monotonically decreasing the entire run, no overfitting
  at any point:
  | step | epoch | eval_loss |
  |---|---|---|
  | 200 | 0.31 | 0.2818 |
  | 400 | 0.62 | 0.2522 |
  | 600 | 0.94 | 0.2354 |
  | 800 | 1.25 | 0.2242 |
  | 1000 | 1.56 | 0.2162 |
  | 1200 | 1.87 | 0.2111 |
  | 1400 | 2.18 | 0.2068 |
  | 1600 | 2.50 | 0.2045 |
  | 1800 | 2.81 | 0.2035 |
- **Artifacts saved**: `checkpoints/sft_qlora_5632/final/` — LoRA adapter
  (`adapter_model.safetensors`, 73.9 MB; `adapter_config.json`: r=16,
  alpha=32, target modules q/k/v/o/gate/up/down_proj, base model
  `Qwen/Qwen2.5-1.5B-Instruct`) plus the full tokenizer (`tokenizer.json`,
  `vocab.json`, `merges.txt`, etc.) — everything an eval/inference script
  needs to load the fine-tuned model. Also `checkpoint-1920/` (same adapter
  plus optimizer/scheduler/rng state, for reference/further training).

### Robustness confirmed during the run
- Resume-crash bug (torch 2.6 `weights_only` vs. transformers 4.46.3's
  `rng_state.pth` load — see DECISIONS.md 2026-09-06) was fixed and verified
  *before* this run started; never needed in practice since the process ran
  uninterrupted end to end, but the safety net was proven working.
- Two separate internet outages during the run (2026-09-07, 2026-09-08 per
  earlier health-check notes) caused only benign PEFT checkpoint-save
  warnings (HEAD request to huggingface.co failing — seen as both DNS
  failures and, once, an `SSLCertVerificationError` — always caught and
  logged as a harmless "silently ignoring the lookup" warning). Training
  never paused or lost progress either time.
- Hourly health checks across the full ~4 day 23 hour run found no other
  issues: no OOM, no NaN loss, disk usage flat at ~321-407M the entire time
  (`save_total_limit=3` correctly capping rolling checkpoints).

### Not yet done
Eval (JSON-validity / quality scoring against the fine-tuned adapter) has not
been started — this entry only confirms training completed and artifacts are
in place. Next step for Day 4-5 is running the eval harness against
`checkpoints/sft_qlora_5632/final`.

---

## 2026-09-06 (cont'd) — intermediate max_seq_len search (4608/5120/5632) + VRAM-saturation mechanism

Continuation of the same-day max_seq_len diagnostic below. Motivation: the
6144 calibration's 4.56x slowdown for a 2x sequence-length increase (562 vs
123.374 s/step) looked disproportionate to pure compute cost, and 4096 was
already known (below) to skew retention away from 7-8 room plans. Tested
whether an intermediate cap does better on both axes at once.

### Retention by room count at 4608 / 5120 / 5632 (reusing cached per-example token counts — no retokenization)
| max_seq_len | rooms=5 (n=1224) | rooms=6 (n=5254) | rooms=7 (n=2071) | rooms=8 (n=2239) | overall |
|---|---|---|---|---|---|
| 4096 (for reference) | 98.4% | 80.7% | 38.2% | 5.9% | 59.01% |
| 4608 | 99.7% | 95.3% | 68.9% | 28.2% | 76.80% |
| **5120** | **99.9%** | **98.9%** | **87.1%** | **58.7%** | **88.41%** |
| 5632 | 99.9% | 99.4% | 95.8% | 81.4% | 95.05% |
| 6144 (for reference) | 99.9% | 99.8% | 98.5% | 92.0% | 97.91% |

**Chose 5120** — smallest of the three that clears both: majority of 7-room
plans (87.1%) and a majority (not just "meaningful share") of 8-room plans
(58.7%). 4608 fails the 8-room bar badly (28.2%, under a third) — it would
reproduce the same undertraining-on-complex-plans failure mode already flagged
for 4096, just less severely.

### max_seq_len=5120 calibration — measured
Ran `--calibrate 20 --max-seq-len 5120`, `output_dir=checkpoints/sft_qlora_diag5120`,
batch/grad-accum unchanged (per-device-batch-size=1, grad-accum=16) — **flagged
before running**: batch=1 is already the floor (2026-08-31/2026-09-05 decisions),
so "reduce micro-batch, raise grad-accum to compensate" had no room to execute;
grad-accum only changes effective batch size / step count, not peak VRAM.

Killed after 2 forced wall-clock checkpoints (same incidental-checkpoint method
as the 6144 run, still not fixed in `train_qlora.py`) rather than waiting the
full 20 steps:
- checkpoint-4 -> checkpoint-8 (clean 4-step delta): 907.967 s / 4 = **226.99
  s/step**.
- **Peak VRAM: 5796 / 6141 MiB = 94.4%** — plateaued almost immediately after
  model load, not a brief spike.
- Kept 9538/10788 train (11.6% dropped, matches the retention table above),
  597 steps/epoch, 1791 total optimiser steps for 3 epochs.
- **Full 3-epoch projection: 1791 x 226.99 s = ~112.9 hours (~4.7 days)**.

**Target VRAM (75-85%) not achieved — flagging as a real constraint, not a
missed optimization.** 94.4% is numerically *higher* than 6144's 93% peak,
despite the shorter cap. Batch=1 was already the floor going in, so there was
no batch/grad-accum lever left to pull toward the target; this result confirms
that limitation rather than revealing a new one.

### max_seq_len=5632 calibration — measured (fills the gap between 5120 and 6144)
Same method: `--calibrate 20 --max-seq-len 5632`, batch=1/grad-accum=16
unchanged (confirmed no lever left, same as 5120). Killed after checkpoint-4 ->
checkpoint-8 (clean 4-step delta): 1035.37 s / 4 = **258.84 s/step**.
- **Peak VRAM: 5864 / 6141 MiB = 95.5%** — the highest of the three intermediate
  points, though see the caveat below on what "peak" means at batch=1.
- Kept 10,254/10,788 (95.05%, matches the earlier retention table), 641
  steps/epoch, 1,923 total optimiser steps for 3 epochs.
- **Full 3-epoch projection: 1923 x 258.84 s = ~138.3 hours (~5.76 days)**.

### Runtime comparison across all measured max_seq_len options
| max_seq_len | kept (train) | s/step | total steps (3 ep) | full-run projection |
|---|---|---|---|---|
| 3072 | 1,235 (11.4%) | 123.374 | 234 | ~8:25-8:58 (h:mm) |
| **5120** | **9,538 (88.4%)** | **226.99** | **1,791** | **~112.9 h (~4.7 days)** |
| **5632** | **10,254 (95.05%)** | **258.84** | **1,923** | **~138.3 h (~5.76 days)** |
| 6144 | 10,563 (97.9%) | 562.0 | 1,983 | ~309.6 h (~12.9 days) |

5120 -> 5632 (+512 tokens, +10% length): step time rose only ~14% (227 -> 259
s/step) — mild, consistent with near-linear scaling. 5632 -> 6144 (also +512
tokens, another +9% length): step time **more than doubled** (259 -> 562
s/step). The steep jump is confined to this second, equally-sized token
window, not spread gradually across 5120-6144 — i.e. the "something
non-linear" threshold sits specifically between 5632 and 6144, not before it.

5120 is ~12.9x longer than 3072 but ~2.7x shorter than 6144; 5632 sits
between them at ~15.6x longer than 3072 and ~2.24x shorter than 6144 — not
merely interpolated, an independently measured point confirming the shape of
the curve.

### VRAM-saturation hypothesis: NOT confirmed by the 5120 result — revising the earlier claim
The previous entry (2026-09-06, below) hypothesized that 6144's disproportionate
slowdown (4.56x time for 2x length) was driven by near-ceiling VRAM triggering
allocator overhead on top of pure compute cost, and expected 5120 — a shorter
cap — to show both lower VRAM and a smaller-than-6144 slowdown factor if that
were true.

**5120 falsifies the clean version of that hypothesis**: its peak VRAM (94.4%)
was *not* lower than 6144's (93%) — essentially the same ceiling-adjacent
level, or even marginally higher — yet its per-step time (227s) is far below
what a naive linear interpolation between the 3072 and 6144 datapoints would
predict at 5120 (~416 s/step, using the slope between those two measured
points). Fitting a pure power law to each adjacent pair instead:
- 3072 -> 5120 (1.667x length): 1.84x slower -> exponent ~1.19 (close to linear).
- 3072 -> 6144 (2x length): 4.56x slower -> exponent ~2.20 (well above linear).

Since VRAM saturation is roughly flat across 5120 and 6144 (93-94% either way)
but the slowdown exponent is not, VRAM headroom alone does not explain the
gap — something specific to the 6144 datapoint (or to crossing some threshold
between 5120 and 6144) is responsible, not simply "how full the card is."

### 5632 result: pins the threshold to a narrow window, and further undercuts the VRAM story
3072 -> 5632 (length ratio 1.833x): 258.84 / 123.374 = 2.098x slower ->
**exponent = ln(2.098) / ln(1.833) = 1.222**. That's close to the 3072->5120
exponent (~1.19, near-linear) and nowhere near the 3072->6144 exponent
(~2.20). **5632 sits squarely on the near-linear trend.** Combined with the
per-window analysis above (mild +14% at 5120->5632, then +117% at
5632->6144, over two equally-sized 512-token windows), the threshold is now
localized tightly: **something changes specifically between 5632 and 6144**,
not gradually across the whole 5120-6144 range.

This also further undercuts the VRAM-saturation hypothesis rather than
supporting it: peak VRAM was 94.4% (5120) -> 95.5% (5632) -> 93% (6144) —
**not monotonic** with length, yet step time is roughly flat then jumps
sharply. If VRAM saturation were the driver, 5632 (the highest measured peak
of the three) should show the worst slowdown, not 6144 (numerically the
*lowest* peak of the three). It doesn't. **Caveat on all these VRAM
percentages**: per-device-batch-size=1 means there's no padding within a
micro-batch, so the instantaneous VRAM reading depends on which specific
example (its exact token length, not the max_seq_len cap) happened to be
active when the poller sampled — these are point-samples of a fluctuating
signal, not a stable ceiling per cap. That noise alone could plausibly explain
small reorderings like 95.5% vs 93%, but it does not explain a 2.2x jump in
step time with no corresponding VRAM jump.

**Leading candidate now**: a kernel-selection or attention-implementation
threshold specific to sequence length (e.g. SDPA switching code paths, a
block/tile-size boundary, or a similar length-bucketed dispatch decision)
somewhere in (5632, 6144]. Thermal throttling was a candidate in the previous
entry but is weakened by 5632's data: its run was almost as data-hungry
(~35 min to 2 forced checkpoints) as 5120's, yet showed none of 6144's
slowdown — if throttling accumulated with wall-clock GPU load, 5632 should
show some drift toward 6144's regime and doesn't. **Not resolved** — pinning
the exact mechanism would need testing inside the (5632, 6144] window (e.g.
5888) and/or instrumenting the attention kernel dispatch directly; out of
scope for a report-only diagnostic. Reporting the localized threshold and the
weakened VRAM/thermal explanations rather than guessing further.

---

## 2026-09-06 — Day 4-5 continued: max_seq_len diagnostic (percentiles, room-count breakdown, 6144 calibration, compact-JSON / grad-checkpoint checks)

Follow-up to 2026-09-05's drop-rate finding. No code changes this entry — pure
measurement, per instructions ("don't change anything yet, just report").

### chars/token ratio confirmed: ~1.6-1.7, not the 2.8 heuristic (closes the open item from 2026-09-05)
Exact-token pass over the real Qwen2.5-1.5B-Instruct tokenizer on the full train
split (n=10,788), using the same `build_messages`/`build_completion_text` as
`prepare_sft_data.py`/`train_qlora.py`: whole-example chars/token ratio p50=1.658,
p90=1.745, p99=1.815, mean=1.70. Confirms the 2026-08-31 heuristic's root cause
exactly: coordinate-heavy JSON runs ~1.6-1.8 chars/token under this BPE
tokenizer, not ~2.8.

### Token-length percentiles by room count (in-scope count, excludes balcony — matches `convert.py`'s 5-8 filter)
| rooms | n | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 5 | 1224 | 2921 | 3183 | 3471 | 3691 | 4252 | 11364 |
| 6 | 5254 | 3641 | 3993 | 4360 | 4588 | 5192 | 15560 |
| 7 | 2071 | 4270 | 4739 | 5239 | 5560 | 6307 | 24533 |
| 8 | 2239 | 4979 | 5435 | 5986 | 6429 | 7178 | 15877 |

Clean monotonic room-count -> length relationship (median 2921 -> 4979 tokens,
5 to 8 rooms), confirming the user's hypothesis that length correlates with room
count. But every room count also has a fat outlier tail (max is 3-6x its own
median), so length is driven by *both* room count and per-plan outliers
(plausibly jagged contours) — not room count alone.

### Retention by candidate max_seq_len (whole train split, n=10,788)
| max_seq_len | retained | dropped |
|---|---|---|
| 3072 (current) | 11.45% | 88.55% |
| 4096 | 59.01% | 40.99% |
| 6144 | 97.91% | 2.09% |
| 8192 | 99.82% | 0.18% |

At 4096 the aggregate 59% retention hides a skew toward the simple end of scope
(consistent with the by-room-count table above: 7- and 8-room medians are
already above 4096) — flagged by the user as the reason 3072/4096 shouldn't be
accepted by default.

### Two "free win" checks before committing to a larger max_seq_len — both came back negative
1. **Completion JSON serialization**: already maximally compact. `plan_prompt.py`'s
   `_COMPACT = dict(separators=(",", ":"), ensure_ascii=False)` is used for both
   `build_messages` (prompt) and `build_completion_text` (completion) — confirmed
   in code and by spot-checking 3 raw records in `sft/data/train.jsonl` (no
   newlines, no space after `:` or `,`). **No token savings available from
   serialization** — nothing to switch.
2. **Gradient checkpointing**: already enabled, at both levels —
   `prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)` and
   `TrainingArguments(gradient_checkpointing=True, gradient_checkpointing_kwargs=
   {"use_reentrant": False})` in `train_qlora.py`. **Already on** — there was no
   "enable it and see if it helps" experiment to run; it was on for the
   2026-09-05 3072 calibration too.

Neither lever was actually available to pull, so per the user's own conditional
plan, no code change and no new calibration run were needed to test them.

### max_seq_len=6144 calibration: killed early, but precise step time recovered anyway
Started `--calibrate 20 --max-seq-len 6144` (separate `output_dir`:
`checkpoints/sft_qlora_diag6144`, so as not to overwrite the 2026-09-05 3072
calibration.json). After 63.5 min of 100%-GPU-utilized wall time with zero
steps logged (logging_steps=10 hadn't fired), the user had it killed —
"we have enough signal."

Two dataloader-worker child processes (`dataloader_num_workers=2`) survived the
first `kill` (parent SIGTERM doesn't propagate to forked workers) and needed a
follow-up `kill -9`; GPU was confirmed back to idle (729 MiB) afterward.

**Incidental bug found**: `WallClockCheckpoint`'s `on_step_end` forces
`control.should_save = True` on the wall-clock interval unconditionally — it
does not check `cfg.calibrate`. So even though `save_strategy="no"` during
calibration and the script prints "no checkpoint written" at the end, three
checkpoints (steps 2, 4, 6) were actually written to disk mid-run. Not fixed
here (not asked, and out of scope for a report-only diagnostic) — flagging for
whoever next touches `train_qlora.py`.

That incidental bug is what let us recover a **precise** step time without
re-running calibration: the three forced checkpoints have file mtimes, and two
consecutive 2-step deltas between them agree closely:
- checkpoint-2 -> checkpoint-4: 1126.84 s / 2 steps = 563.4 s/step
- checkpoint-4 -> checkpoint-6: 1121.37 s / 2 steps = 560.7 s/step
- **~562 s/step**, consistent between both intervals.

Peak VRAM during the run: **5730 / 6141 MiB (93%)** — tight (vs. ~500-600 MB
headroom at 3072), no OOM.

**562 s/step is ~4.56x slower than 3072's 123.374 s/step** — worse than the
~2x a naive linear-in-sequence-length model would predict for doubling
`max_seq_len`. Likely compounding of (a) attention cost scaling worse than
linear in sequence length and (b) sitting at 93% VRAM utilization (vs. ~90%
at 3072), which plausibly adds allocator/fragmentation overhead on top of the
raw compute cost. Not isolated further — would need a sweep at intermediate
lengths (e.g. 4096, 5120) to separate the two effects, not done here.

**Projected full run at 6144**: 1983 optimiser steps (661/epoch x 3 epochs) x
562 s/step = ~309.6 hours (~12.9 days) — not the "longer but acceptable"
runtime the user was willing to trade for better room-count coverage; this is
off by roughly two orders of magnitude from anything practical locally, even
with a generous tolerance.

### Open decision (unchanged from 2026-09-05, now with better data)
No max_seq_len chosen yet. 3072 undertrains on room count alone; 4096 skews
away from 7-8 room plans (the complex end of the project's own 5-8 room scope);
6144 retains well (97.91%) but is not locally feasible at ~562 s/step. Options
not yet evaluated: an intermediate max_seq_len between 4096 and 6144 (untested),
stratified subsampling to rebalance room-count representation within a smaller
cap, or moving the real run to Colab's T4 (unverified whether the T4 handles
the longer-sequence slowdown better or worse than this GPU — the original ~1hr
heuristic was for 3072, not 6144). Not decided here.

---

## 2026-09-05 — Day 4-5 continued: local run (WSL/RTX 4050) — data prep + calibration

Environment/dependency reasoning: DECISIONS.md 2026-09-05. No verifier or
`reward.py` touched.

### `python -m data.run` — clean rerun locally
Venv had only `pip` installed (nothing from `sft/requirements.txt` was actually
present). Installed `shapely` only, then the pipeline ran end-to-end with no
code changes needed: **13,485 / 17,000 plans converted**, same split sizes as
the 2026-08-30 run (train 10,788 / val 1,349 / test 1,348), same drop-reason
breakdown. Confirms the pipeline is deterministic and has no hidden Colab-only
dependency.

### `prepare_sft_data.py` on the real Qwen2.5-1.5B-Instruct tokenizer
First run hit `NameError: name 'Extension' is not defined` inside
`transformers.apply_chat_template` — caused by a missing `jinja2` (not listed
in `sft/requirements.txt`; installed 3.1.6). After that, exact token stats:

| split | exact tokens p50/p90/p99/max | dropped @ max_seq_len=3072 |
|---|---|---|
| train | 3897 / 5210 / 6624 / 24533 | 9,553 / 10,788 = **88.55%** |
| val   | 3894 / 5100 / 6695 / 12983 | 1,201 / 1,349 = **89.03%** |

This is far above the 2026-08-31 chars/2.8-heuristic estimate (p50≈2305,
p90≈2965, expected ~8-9% drop). The heuristic badly underestimates
tokens-per-char for coordinate-heavy JSON (digit runs tokenize far less
efficiently than prose). See DECISIONS.md 2026-09-05 for the consequence
(effective train set shrinks to ~1,235 plans) and the open decision.

### `train_qlora.py --calibrate 50` — no OOM, but ~8-9× slower than the T4 heuristic
Ran to completion on the RTX 4050 Laptop (6 GB VRAM), no code changes:
- Peak VRAM ~5.5–5.6 GB / 6.1 GB — tight (~500-600 MB headroom) but stable
  across 50 steps, no OOM.
- **123.374 s/step** median (44 of 50 steps, first 5 skipped). Loss dropped
  0.482 → 0.353 over 50 steps (sane early SFT curve, not diagnostic of
  anything at this step count).
- Tokenizer pass inside the script independently reproduced the drop-rate
  finding: **kept 1,235 / 10,788 train, 35 / 256 eval subset** at
  `max_seq_len=3072`.
- Full 3-epoch run at the filtered size (234 optimiser steps) projects to
  **~8:25–8:58 wall-clock** — vs. the ~1 hr T4 heuristic on record. Calibration
  did its job (caught this before an unattended multi-hour run was launched
  blind); decision on how to proceed is open (DECISIONS.md 2026-09-05).

### Not done this session
Did not proceed to the full training run — per instructions, calibration
output is reported back for confirmation first. Both the drop-rate and the
runtime finding are large enough deviations from the 2026-08-31 plan that they
need a decision, not a silent default.

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

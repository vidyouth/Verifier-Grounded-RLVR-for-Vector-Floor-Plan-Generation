# Project Context — Verifier-Grounded RLVR for Floor-Plan Generation

This is the single source of truth for the project. Upload this alongside EXECUTION_PLAN.md and CLAUDE.md as Project knowledge. Anything not written down here or in those two files should be treated as undecided — ask, don't assume.

## 1. What this project is
An LLM generates floor plans as JSON vector polygons (extending Lara et al. 2026, arXiv:2605.14117), fine-tuned via GRPO with a reward that gates hard on statutory-style clause compliance before rewarding aesthetic/topological quality. The research question is not "can you generate compliant plans" (that already exists) — it's whether a fast, deterministic clause verifier can be trusted as an RL reward signal: does it agree with an independently-built second checker, and can the policy learn to game it (reward hacking / false compliance).

**Locked scope (Phase 3 — do not drift back to the broader Phase 2 draft):**
- Single-floor residential layouts, 5–8 rooms.
- Exactly 3 clause families: minimum corridor clear width, minimum door clear width, maximum egress travel distance.
- One fixed jurisdiction/edition — but the actual threshold numbers are a generic/placeholder code, not a real jurisdiction (India NBC, IBC, etc.), because grounding in a real code wasn't feasible solo in 8 days. This must be stated as a limitation everywhere, not silently treated as real.
- Explicitly out of scope: setbacks/FAR (needs plot-boundary data RPLAN doesn't have), daylight/ventilation, digital-twin/3D simulation validation, the compliance-realism "strictness dial" trade-off curve, multi-floor or non-residential plans.

## 2. Technical spec (from the Phase 3 draft)

**Representation**: JSON schema per plan — metadata (schema version, scale, jurisdiction/edition, occupancy), boundary (exterior polygon + entrance), rooms (id, type, polygon, target area), openings (door/window, host wall, endpoints, clear width, swing), relations (requested + derived adjacency), audit (clause id, measured value, threshold, tolerance, status). Coordinates in meters, local Cartesian frame.

**Verifier bank, staged**: syntax/schema gate → geometry gate (simple/non-overlapping polygons, valid openings) → topology (reachability from every room to entrance) → statutory (the 3 clauses) → soft quality. Each verifier ships with its own test suite: canonical pass/fail, near-threshold both sides, mutation-based adversarial cases. Nothing becomes a training reward until its test suite passes.

**Reward function** (feasibility-first — this is the core mechanism, don't let it get "softened" into a weighted-sum-only reward):
- Margin: m_k(y) = (g_k(y) − τ_k) / max(|τ_k|, ε) for minimum requirements, sign-flipped for maximum requirements.
- Clause satisfaction c_k(y) = 1[m_k(y) ≥ 0], violation severity s_k(y) = max(0, −m_k(y)).
- Statutory score: R_code(y) = Σ w_k c_k(y) − λ Σ w_k s_k(y).
- Total reward: R(y) = −1 if the syntax/geometry gate fails; otherwise R(y) = M·C_all(y) + R_code(y) + β·Q(y), where C_all is 1 only if every clause passes and M dominates the max possible quality-score spread — so an attractive violating plan can never outrank a compliant one.
- GRPO: group-relative standardized advantage, clipped policy-ratio objective with KL regularization to the SFT reference policy.

**Baselines to compare against**: SFT-only; proxy-only RLVR (Lara et al. reproduction — area/overlap/connectivity reward only); the proposed statutory RLVR; inference-time filtering using the same checker; post-hoc repair modeled on Design Healing.

**Independent validation**: a second checker, separately implemented, never used during training, run only at eval time. Disagreements with the training-time verifier get categorized (false pass, false rejection, schema ambiguity, reference-checker defect).

## 3. Working constraints
- Solo project, 8-day timeline (see EXECUTION_PLAN.md for the day-by-day split).
- Compute: Colab free tier only (T4, ~12hr session cap, disconnects on idle). Training scripts must be checkpoint/resume-safe. A 7–8B QLoRA run may not be realistic on this hardware — a 1–3B base model is the fallback if Day 4–5 shows the T4 can't keep up; this decision should be made explicitly and logged in DECISIONS.md, not silently downgraded.
- Dev split: VS Code + Claude Code for everything except GPU training; Colab notebooks for SFT/GRPO runs.
- Two work lanes: fast lane (data prep, boilerplate, eval plumbing) — Claude Code just writes it. Slow lane (verifier math, reward function, GRPO loop) — Claude Code explains the approach before finalizing, since these are the parts that need to be defensible in a viva/writeup.

## 4. Documentation requirements (Claude Code maintains these, not just this chat)
- /docs/FINDINGS.md — dated entries: what was tried, results, what broke and how it was fixed.
- /docs/DECISIONS.md — dated entries: every design choice and the reasoning (base model size, threshold values, split strategy, etc.).
- /docs/LIMITATIONS.md — living list; first entry always: thresholds are a placeholder/generic code, not a real jurisdiction, so results are not evidence of real-world compliance.

## 5. How this Project chat should operate
When the user says something like "let's do day N," this chat should:
1. Pull that day's scope from EXECUTION_PLAN.md.
2. Ask any clarifying questions needed to remove ambiguity before generating a prompt — e.g., which base model (once decided, stop re-asking), any change in scope, anything Day N depends on from a prior day that isn't confirmed done.
3. Output one ready-to-paste prompt for Claude Code that (a) states the day's deliverable, (b) references CLAUDE.md conventions (fast/slow lane, checkpointing, doc updates), (c) tells Claude Code to update FINDINGS.md/DECISIONS.md/LIMITATIONS.md as it goes, and (d) explicitly asks Claude Code to surface any doubts or ambiguous decisions back to the user rather than guessing silently.
4. Do not silently expand scope back toward the Phase 2 draft (setbacks, daylight, digital twin, strictness dial) even if it would make the paper more impressive — that was explicitly rejected for feasibility reasons.

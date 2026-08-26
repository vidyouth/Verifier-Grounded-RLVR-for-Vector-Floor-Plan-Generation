Execution Plan — Verifier-Grounded RLVR for Floor-Plan Generation
Setup
Solo project.
Compute: Colab free tier (T4 GPU, ~12hr session cap, disconnects on idle).
Codebase: written by Claude Code. Fast lane for boilerplate/data-prep/eval plumbing — Claude Code just writes it. Slow lane for the verifier bank and reward function — Claude Code explains the logic before/while writing it.
Threshold values: generic/placeholder code, not a real jurisdiction (India NBC, IBC, etc.). This must be labeled explicitly everywhere — in code comments, in the paper, in every doc — since it directly limits how strong any "compliance" claim can be.
Scope (Phase 3, locked in): single-floor residential, 5–8 rooms, 3 clauses — minimum corridor clear width, minimum door clear width, maximum egress travel distance.
Colab free-tier reality check
A 7–8B model won't train in one sitting on a free T4. Use a small base model (1–3B) with 4-bit QLoRA, or accept multiple short sessions.
Checkpoint to Google Drive every N steps — a disconnect should never cost more than ~15–20 minutes of progress.
Every training script must be restart-safe: check for the last checkpoint on launch and resume, never assume an uninterrupted run.
Repo structure
/data              RPLAN-derived JSON, normalized, splits (train/val/test by parent building)
/verifiers         geometry_gate.py, topology.py, corridor.py, door.py, egress.py
/verifiers/tests   unit + near-boundary + mutation-based test cases per verifier
/sft               SFT training script, data prep
/rlvr              GRPO training script, reward function
/eval              independent second checker (separate code path), eval scripts
/docs
  FINDINGS.md      dated log — what was tried, what worked/broke, numbers
  DECISIONS.md     dated log — every design choice and why
  LIMITATIONS.md   living list, starting with the placeholder-code caveat
CLAUDE.md
8-day schedule

Day 1 — Verifier bank v1: geometry gate, corridor checker, door checker. Fast lane write-up; slow-lane walkthrough of the logic at day's end, review test results.

Day 2 — Egress verifier (visibility graph, shortest path to exit) + topology checker + full unit/near-boundary/mutation test suite.

Day 3 — Data prep: pull Lara et al. / RPLAN JSON data, normalize coordinates, dedupe, split 80/10/10 by parent building identity (not by augmented example).

Day 4–5 — SFT: pick small base model, QLoRA fine-tune on Colab, checkpoint/resume wiring, validate JSON-validity rate on held-out set. Budget slack here — free-tier disconnects hurt most during training.

Day 6–7 — GRPO: implement the feasibility-first reward (hard gate on clause satisfaction, then soft quality score), run training, watch for reward collapse or low-variance groups. Expect at least one failed run — log it, don't panic.

Day 8 — Eval: run the independent second checker against training-time verifier outputs, compute compliance / false-compliance rates, finalize LIMITATIONS.md with the placeholder-code caveat front and center.

Documentation (continuous, not end-of-project)
FINDINGS.md — dated entries of what was tried and what happened.
DECISIONS.md — dated entries of design choices and reasoning (base model, thresholds, split strategy, etc.).
LIMITATIONS.md — living list; first entry is always the placeholder-code caveat.
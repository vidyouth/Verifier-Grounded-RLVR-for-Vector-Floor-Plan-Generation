CLAUDE.md
Project

Verifier-grounded RLVR for JSON vector floor-plan generation. Single-floor residential, 5–8 rooms. Three clauses: minimum corridor clear width, minimum door clear width, maximum egress travel distance.

Clause thresholds are a placeholder/generic code, not a real jurisdiction. Always label them as such in code comments, docstrings, and any doc that mentions "compliance."

Environment
Solo project. Colab free tier (T4 GPU, ~12hr session cap, disconnects on idle).
Local dev in VS Code; GPU work happens in Colab notebooks.
All training scripts must be checkpoint/resume-safe — never assume an uninterrupted long run.
Working style — two lanes
Fast lane (default): data prep, boilerplate, eval plumbing, plotting, repo scaffolding. Write it and move on.
Slow lane (pause and explain first): the verifier bank logic (corridor/door/egress math), the reward function (feasibility gate + statutory score + soft quality), and the GRPO loop. For these, give a plain-language explanation of the approach before writing the code, and flag any nontrivial design choice for review before finalizing.
Documentation — maintain continuously
/docs/FINDINGS.md — dated entries: what was tried, results, anything that broke and how it was resolved.
/docs/DECISIONS.md — dated entries: every design choice and the reasoning behind it (model choice, threshold values, split strategy, etc.).
/docs/LIMITATIONS.md — living list. First entry, always present: "clause thresholds are a placeholder/generic code, not a real statutory jurisdiction — results should not be read as evidence of real-world code compliance."

Update these at the end of each work session, not retroactively.

Code conventions
Verifiers are pure functions: JSON plan in, structured result out (clause id, measured value, threshold, margin, pass/fail). No side effects.
Every verifier ships with its own test file: canonical pass, canonical fail, near-threshold on both sides, at least one mutation-based adversarial case.
Training scripts: resumable from last checkpoint, checkpoint interval configurable, defaulting to something safe for a free Colab session (e.g. every 15 minutes or N steps).
Testing

The verifier bank must pass its full test suite before it is used as a training reward signal.
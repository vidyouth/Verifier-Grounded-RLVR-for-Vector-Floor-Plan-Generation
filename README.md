# Verifier-Grounded RLVR for Vector Floor-Plan Generation

Verifier-grounded reinforcement learning with verifiable rewards (RLVR) for JSON
vector floor-plan generation. Single-floor residential, 5–8 rooms. Three clauses:
minimum corridor clear width, minimum door clear width, maximum egress travel
distance.

> **Clause thresholds are a placeholder / generic code, not a real jurisdiction.**
> Results are not evidence of real-world building-code compliance. See
> [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Repository layout

| Path | Contents |
|---|---|
| `verifiers/` | Pure-function verifiers (`geometry_gate`, `topology`, `corridor`, `door`, `egress`) + tests |
| `rlvr/` | Feasibility-first reward function + (later) GRPO training |
| `data/` | ResPlan → project-schema conversion pipeline + tests |
| `sft/` | SFT training / data prep |
| `eval/` | Independent second checker |
| `docs/` | `FINDINGS.md`, `DECISIONS.md`, `LIMITATIONS.md` (updated each session) |

## Data

This project trains and evaluates on **ResPlan**, converted to the project's
floor-plan JSON schema by the pipeline in [`data/`](data/).

> **ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor
> Plans.** The ResPlan Authors, 2025. <https://github.com/m-agour/ResPlan>
>
> Dataset licensed under **Creative Commons Attribution 4.0 International
> (CC BY 4.0)** — <https://creativecommons.org/licenses/by/4.0/>. The ResPlan
> code (`resplan_utils.py`, etc.) is MIT-licensed.

ResPlan is used **unmodified as input** to a schema-conversion pipeline
(`python -m data.run`); the raw dataset files are not redistributed here (see
[`data/raw/PROVENANCE.md`](data/raw/PROVENANCE.md) for download steps). A small
set of converted example plans is included under [`data/samples/`](data/samples/).
Any redistributed derivative of ResPlan would itself carry CC BY 4.0 plus a note
of changes.

This project was originally planned around RPLAN / the Lara et al. data; the
switch to ResPlan (2026-08-30, RPLAN access unresolved within the timeline) and
its consequences for cross-paper comparison are recorded in
[`docs/DECISIONS.md`](docs/DECISIONS.md) and [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Tests

```bash
python -m pytest verifiers/tests/ rlvr/tests/ data/tests/ -q
```

The verifier bank must pass its full suite before it is used as a training
reward signal.

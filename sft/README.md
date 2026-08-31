# SFT — QLoRA fine-tune (Day 4-5)

Supervised fine-tune of **Qwen2.5-1.5B-Instruct** (4-bit NF4 QLoRA) to emit
project-schema floor-plan JSON from a boundary + entrance + room list. This is
the SFT reference policy for the Day 6-7 GRPO run.

> Clause thresholds referenced downstream are a **placeholder / generic code, not
> a real jurisdiction** (`docs/LIMITATIONS.md`).

## Files

| File | Role |
|---|---|
| `plan_prompt.py` | Single source of truth for prompt/completion construction + downstream reassembly (metadata templated, `relations` derived from openings, `audit=[]`). |
| `prepare_sft_data.py` | `data/processed/{train,val,test}.jsonl` → `sft/data/*.jsonl` instruction pairs. Needs only `shapely`. |
| `train_qlora.py` | The training run. Plain script, one command, wall-clock checkpoint to Drive, auto-resume. |
| `eval_json_validity.py` | Headline JSON-validity rate on val (+ `geometry_gate` pass rate, secondary). |
| `requirements.txt` | Pinned Colab deps. |

Prompt = `{boundary:{polygon,entrance}, rooms:[{type,target_area}]}` (target_area
recomputed from each polygon — it is the PROJECT_CONTEXT.md schema field the
ResPlan converter does not emit). Completion = `{boundary, rooms:[{id,type,
polygon}], openings:[…]}`. `metadata` / `relations` / `audit` are **not**
generated — they are boilerplate or fully derivable, so they are stamped on
after generation.

## Colab setup (paste verbatim into a T4 notebook)

```python
# 1. GPU check — must say Tesla T4
!nvidia-smi -L

# 2. Clone the repo
%cd /content
!git clone https://github.com/<you>/Verifier-Grounded-RLVR-for-Vector-Floor-Plan-Generation-.git repo
%cd /content/repo

# 3. Install pinned deps (torch stays as Colab ships it)
!pip -q install -r sft/requirements.txt

# 4. Mount Drive — checkpoints live here so a disconnect costs <= 15 min
from google.colab import drive
drive.mount('/content/drive')

# 5. Build the SFT pairs. --tokenizer makes it print the EXACT over-length
#    drop rate at max_seq_len 3072 (not the char-based estimate).
!python sft/prepare_sft_data.py --splits train val \
    --tokenizer Qwen/Qwen2.5-1.5B-Instruct --max-seq-len 3072

# 6. CALIBRATE FIRST (~50 steps, ~10-20 min): prints a measured full-run time
#    estimate from real T4 throughput, then exits without checkpointing.
#    Report the printed line back before launching the unattended run.
!python sft/train_qlora.py \
    --output-dir /content/drive/MyDrive/floorplan_rlvr/sft_qlora --calibrate 50

# 7. Full run. Re-run this SAME cell after any disconnect — it resumes itself.
!python sft/train_qlora.py \
    --output-dir /content/drive/MyDrive/floorplan_rlvr/sft_qlora
```

The `--calibrate` step exists because the "~1 hr for 3 epochs" figure in
DECISIONS.md is a pre-run heuristic; FLOP math suggests it could be several
hours on a T4. Calibration measures real `s/step` (median, first 5 steps
skipped) and extrapolates. Both numbers stay on record in DECISIONS.md.

Smoke test first (optional, ~3 min):

```python
!python sft/train_qlora.py --output-dir /content/drive/MyDrive/floorplan_rlvr/sft_smoke \
    --limit 128 --max-steps 20 --checkpoint-minutes 2
```

## Eval (after training)

```python
!python sft/eval_json_validity.py \
    --adapter /content/drive/MyDrive/floorplan_rlvr/sft_qlora/final
```

Deterministic greedy decoding over the 1,349 val plans; writes one result line
per plan and resumes if interrupted. Prints the strict `json.loads` rate
(headline) plus the `geometry_gate` pass rate.

## Healthy-run checklist

`train_qlora.py` prints one status line per log step:

```
step   120/1875 | ep 0.64 | loss 0.2413 | lr 1.71e-04 | 2.1s/it | ckpt in  6.4m | elapsed  14.0m | eta  49.3m
```

- **loss** starts ~1-2, should fall below ~0.3 within the first epoch and keep
  drifting down. Flat/NaN loss → stop and check.
- **ckpt in** counts down to the next Drive checkpoint; after a disconnect,
  re-running the command prints `RESUMING from …/checkpoint-N`.
- **s/it** ~1.5-2.5 on a healthy T4 at seq-len 3072.

Hyperparameters and their rationale: `docs/DECISIONS.md`, 2026-08-31.

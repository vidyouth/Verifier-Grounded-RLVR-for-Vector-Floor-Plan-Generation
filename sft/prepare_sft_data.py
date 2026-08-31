"""
SFT data prep (Day 4-5, deliverable 1).

Convert the converted floor-plan splits (data/processed/{train,val,test}.jsonl —
project schema, one plan per line) into instruction/response pairs for QLoRA
fine-tuning of Qwen2.5-1.5B-Instruct.

    python sft/prepare_sft_data.py                          # all splits -> sft/data/
    python sft/prepare_sft_data.py --splits train val
    python sft/prepare_sft_data.py --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
                                   --max-seq-len 3072       # + exact token-length report

Output: sft/data/<split>.jsonl, one JSON object per line:

    {"plan_id": <int>,
     "messages":   [{"role": "system", ...}, {"role": "user", ...}],
     "completion": "<compact plan JSON: boundary + rooms + openings>"}

`messages` is stored un-rendered (no chat template applied) so the tokenizer is
chosen at train time. Rendering + tokenisation + the final max-seq-len filter
happen in train_qlora.py; this script only reports what that filter will do.

Prompt / completion split rationale and the templated-downstream pieces
(metadata, relations, audit) are documented in sft/plan_prompt.py. Clause
thresholds referenced downstream are a PLACEHOLDER / generic code, not a real
jurisdiction (docs/LIMITATIONS.md).

No model, no GPU — needs only `shapely` (and optionally `transformers` for
--tokenizer). Safe to run locally; re-run in Colab before training if sft/data/
was not carried over (it is derived, not committed).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sft.plan_prompt import build_completion_text, build_messages  # noqa: E402

_PROCESSED = _REPO_ROOT / "data" / "processed"
_OUT_DIR = _REPO_ROOT / "sft" / "data"

# Rough chars-per-token for compact coordinate-heavy JSON under the Qwen2.5 BPE
# tokenizer (digits and '.' split fine-grained). Measured ~2.8 on a sample; used
# only for the estimate printed when --tokenizer is not given.
_CHARS_PER_TOKEN = 2.8


def _iter_plans(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _plan_id(plan: dict) -> int:
    return int(plan.get("metadata", {}).get("source", {}).get("plan_id", -1))


def build_split(split: str, tokenizer=None, max_seq_len: int = 3072) -> dict:
    src = _PROCESSED / f"{split}.jsonl"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — run `python -m data.run` first (Day 3 pipeline)."
        )
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"{split}.jsonl"

    n = 0
    prompt_chars: list[int] = []
    completion_chars: list[int] = []
    exact_over = 0
    exact_tok: list[int] = []

    with out_path.open("w", encoding="utf-8") as out:
        for plan in _iter_plans(src):
            messages = build_messages(plan)
            completion = build_completion_text(plan)
            rec = {"plan_id": _plan_id(plan), "messages": messages, "completion": completion}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

            p_chars = len(messages[0]["content"]) + len(messages[1]["content"])
            prompt_chars.append(p_chars)
            completion_chars.append(len(completion))

            if tokenizer is not None:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                n_prompt = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
                n_completion = len(tokenizer(completion, add_special_tokens=False)["input_ids"])
                total = n_prompt + n_completion + 1  # + EOS
                exact_tok.append(total)
                if total > max_seq_len:
                    exact_over += 1

    def _pct(xs: list[int], q: float) -> int:
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * q))]

    stats = {
        "split": split,
        "pairs_written": n,
        "out_path": str(out_path.relative_to(_REPO_ROOT)),
        "prompt_chars": {
            "p50": _pct(prompt_chars, 0.50),
            "p90": _pct(prompt_chars, 0.90),
            "max": max(prompt_chars),
        },
        "completion_chars": {
            "p50": _pct(completion_chars, 0.50),
            "p90": _pct(completion_chars, 0.90),
            "p99": _pct(completion_chars, 0.99),
            "max": max(completion_chars),
        },
    }
    if tokenizer is not None:
        stats["exact_tokens"] = {
            "p50": _pct(exact_tok, 0.50),
            "p90": _pct(exact_tok, 0.90),
            "p99": _pct(exact_tok, 0.99),
            "max": max(exact_tok),
            "mean": round(statistics.mean(exact_tok), 1),
        }
        stats["dropped_over_max_seq_len"] = {
            "max_seq_len": max_seq_len,
            "count": exact_over,
            "pct": round(100 * exact_over / n, 2),
        }
    else:
        est_total = [
            round((pc + cc) / _CHARS_PER_TOKEN) for pc, cc in zip(prompt_chars, completion_chars)
        ]
        stats["estimated_tokens_chars_over_%.1f" % _CHARS_PER_TOKEN] = {
            "p50": _pct(est_total, 0.50),
            "p90": _pct(est_total, 0.90),
            "p99": _pct(est_total, 0.99),
            "max": max(est_total),
            "note": "estimate only — pass --tokenizer for the exact figure the "
                    "train-time filter uses",
        }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="which processed splits to convert (default: all three)")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer id/path for an exact token-length report "
                         "(e.g. Qwen/Qwen2.5-1.5B-Instruct). Optional.")
    ap.add_argument("--max-seq-len", type=int, default=3072,
                    help="sequence-length budget the train-time filter will use "
                         "(default: 3072). Only affects the report here.")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"reading   {_PROCESSED}/<split>.jsonl")
    print(f"writing   {_OUT_DIR}/<split>.jsonl")
    print(f"tokenizer {args.tokenizer or '(none — char-based estimate)'}")
    print()
    for split in args.splits:
        stats = build_split(split, tokenizer=tok, max_seq_len=args.max_seq_len)
        print(json.dumps(stats, indent=2))
        print()


if __name__ == "__main__":
    main()

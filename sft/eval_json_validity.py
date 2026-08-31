r"""
JSON-validity eval on the held-out val split (Day 4-5, deliverable 3).

Headline metric (per EXECUTION_PLAN.md Day 4-5): the strict JSON-parseable rate
of the SFT model's generations on data/processed/val.jsonl (1,349 plans) — the
fraction whose raw completion is accepted by `json.loads` untouched.

Secondary, non-blocking: of those generations, the fraction that also pass
`geometry_gate` (schema/polygon validity) once reassembled into a full plan
(templated metadata + relations derived from openings — see sft/plan_prompt.py).
This is NOT the full clause-compliance eval — that is Day 6-7/8.

    python sft/eval_json_validity.py \
        --adapter /content/drive/MyDrive/floorplan_rlvr/sft_qlora/final

Deterministic greedy decoding. Writes one result line per plan to --out as it
goes; re-running skips plans already scored (resume-safe for a Colab disconnect).
Clause thresholds referenced downstream are a PLACEHOLDER / generic code, not a
real jurisdiction (docs/LIMITATIONS.md).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sft.plan_prompt import build_prompt_text, reconstruct_plan  # noqa: E402
from verifiers import geometry_gate  # noqa: E402

_VAL = _REPO_ROOT / "data" / "processed" / "val.jsonl"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _plan_id(plan: dict) -> int:
    return int(plan.get("metadata", {}).get("source", {}).get("plan_id", -1))


def _load_val(limit: int = 0) -> list[dict]:
    rows = []
    with _VAL.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _score_generation(text: str) -> dict:
    """strict json.loads on the raw text; diagnostics for fence-stripped + geometry_gate."""
    raw = text.strip()
    res = {"valid_json": False, "valid_after_fence_strip": False,
           "geometry_gate_pass": False, "note": ""}

    obj = None
    try:
        obj = json.loads(raw)
        res["valid_json"] = True
        res["valid_after_fence_strip"] = True
    except json.JSONDecodeError:
        stripped = _FENCE.sub("", raw).strip()
        try:
            obj = json.loads(stripped)
            res["valid_after_fence_strip"] = True
        except json.JSONDecodeError:
            res["note"] = "unparseable"
            return res

    try:
        plan = reconstruct_plan(obj)
        res["geometry_gate_pass"] = bool(geometry_gate.check(plan)["pass"])
    except (KeyError, TypeError, ValueError) as e:
        res["note"] = f"reconstruct/gate error: {type(e).__name__}"
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="path to the trained LoRA adapter (…/final)")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default=None, help="results JSONL (default: <adapter>/../eval_json_validity.jsonl)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=3200)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N val plans (dev)")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter = Path(args.adapter)
    out_path = Path(args.out) if args.out else adapter.parent / "eval_json_validity.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    done.add(int(json.loads(line)["plan_id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"resume: {len(done)} plans already scored in {out_path}")

    rows = _load_val(limit=args.limit)
    todo = [p for p in rows if _plan_id(p) not in done]
    print(f"val plans: {len(rows)}  |  to generate: {len(todo)}")
    if not todo:
        _summarise(out_path)
        return

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, torch_dtype=torch.float16,
        attn_implementation="sdpa", device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()

    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as out:
        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i:i + args.batch_size]
            prompts = [build_prompt_text(p, tok) for p in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, do_sample=False, num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                )
            new = gen[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(new, skip_special_tokens=True)
            for plan, text in zip(chunk, texts):
                rec = {"plan_id": _plan_id(plan), **_score_generation(text)}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n_done = min(i + args.batch_size, len(todo))
            rate = (time.time() - t0) / n_done
            print(f"  {n_done:>4}/{len(todo)}  ({rate:4.1f}s/plan, "
                  f"eta {rate*(len(todo)-n_done)/60:5.1f}m)", flush=True)

    _summarise(out_path)


def _summarise(out_path: Path) -> None:
    recs = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(recs)
    if not n:
        print("no results to summarise.")
        return
    valid = sum(r["valid_json"] for r in recs)
    fence = sum(r["valid_after_fence_strip"] for r in recs)
    geo = sum(r["geometry_gate_pass"] for r in recs)
    summary = {
        "n_scored": n,
        "json_validity_rate": round(valid / n, 4),                 # ← headline (strict)
        "valid_after_fence_strip_rate": round(fence / n, 4),       # diagnostic
        "geometry_gate_pass_rate_over_all": round(geo / n, 4),     # secondary
        "geometry_gate_pass_rate_over_valid": round(geo / valid, 4) if valid else 0.0,
    }
    (out_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 60)
    print(f"  JSON-validity rate (headline, strict) : {summary['json_validity_rate']:.4f}  "
          f"({valid}/{n})")
    print(f"  after code-fence strip (diagnostic)   : {summary['valid_after_fence_strip_rate']:.4f}")
    print(f"  geometry_gate pass / all (secondary)  : {summary['geometry_gate_pass_rate_over_all']:.4f}  "
          f"({geo}/{n})")
    print(f"  geometry_gate pass / valid-JSON       : {summary['geometry_gate_pass_rate_over_valid']:.4f}")
    print("=" * 60)
    print(f"  summary -> {out_path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()

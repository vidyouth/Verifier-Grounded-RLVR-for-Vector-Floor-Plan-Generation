r"""
QLoRA SFT — Qwen2.5-1.5B-Instruct on ResPlan-converted floor plans (Day 4-5, deliverable 2).

Plain script: kick off with one command and leave it alone. Console logging is a
single status line per log step (step / loss / lr / it-speed / time-to-next-
checkpoint / elapsed) so a glance tells you it is healthy. Checkpoints to Google
Drive on a wall-clock interval (default 15 min); relaunching the exact same
command auto-resumes from the latest checkpoint with no manual step.

────────────────────────────────────────────────────────────────────────────────
COLAB SETUP (paste verbatim into a T4 notebook)
────────────────────────────────────────────────────────────────────────────────
    # 1. GPU check — must say Tesla T4
    !nvidia-smi -L

    # 2. Clone the repo
    %cd /content
    !git clone https://github.com/<you>/Verifier-Grounded-RLVR-for-Vector-Floor-Plan-Generation-.git repo
    %cd /content/repo

    # 3. Install pinned deps (see sft/requirements.txt for versions + rationale)
    !pip -q install -r sft/requirements.txt

    # 4. Mount Drive (checkpoints survive a disconnect here)
    from google.colab import drive
    drive.mount('/content/drive')

    # 5. Build the SFT pairs (needs only shapely; ~30 s). Prints the EXACT
    #    over-length drop rate at max_seq_len 3072 on the real tokenizer.
    !python sft/prepare_sft_data.py --splits train val \
        --tokenizer Qwen/Qwen2.5-1.5B-Instruct --max-seq-len 3072

    # 6. CALIBRATE FIRST — ~50 steps, prints a measured full-run estimate, exits.
    #    Report the printed number back before launching the unattended run.
    !python sft/train_qlora.py \
        --output-dir /content/drive/MyDrive/floorplan_rlvr/sft_qlora --calibrate 50

    # 7. Full run. Re-run this SAME cell after any disconnect — it resumes itself.
    !python sft/train_qlora.py \
        --output-dir /content/drive/MyDrive/floorplan_rlvr/sft_qlora

    # Smoke test (optional, ~3 min):
    # !python sft/train_qlora.py --output-dir /content/drive/MyDrive/floorplan_rlvr/sft_smoke \
    #     --limit 128 --max-steps 20 --checkpoint-minutes 2
────────────────────────────────────────────────────────────────────────────────

Hyperparameters and their reasoning are logged in docs/DECISIONS.md
(2026-08-31, Day 4-5). Clause thresholds referenced anywhere downstream are a
PLACEHOLDER / generic code, not a real jurisdiction (docs/LIMITATIONS.md).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DATA_DIR = _REPO_ROOT / "sft" / "data"


# ── Config (defaults = the proposed, DECISIONS-logged setup) ─────────────────

@dataclass
class Config:
    # base model / quantisation
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_seq_len: int = 3072

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )

    # optimisation
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    per_device_batch_size: int = 1
    grad_accum: int = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    seed: int = 42

    # checkpoint / logging
    output_dir: str = "/content/drive/MyDrive/floorplan_rlvr/sft_qlora"
    checkpoint_minutes: float = 15.0
    save_steps_floor: int = 200
    save_total_limit: int = 3
    logging_steps: int = 10
    eval_steps: int = 200
    eval_subset: int = 256

    # dev knobs
    limit: int = 0          # >0: use only the first N train pairs
    max_steps: int = -1     # >0: hard cap on optimiser steps (smoke tests)
    calibrate: int = 0      # >0: run this many steps, print a measured full-run
                            #     time estimate, then exit WITHOUT a long run


# ── Dataset ─────────────────────────────────────────────────────────────────

def _load_pairs(path: Path, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _tokenize_pairs(rows: list[dict], tokenizer, max_seq_len: int) -> tuple[list[dict], int]:
    """
    Each row -> {"input_ids", "labels", "length"}.
    input_ids = prompt_ids + completion_ids + [eos]
    labels    = [-100]*len(prompt_ids) + completion_ids + [eos]   (loss on completion only)
    Rows exceeding max_seq_len are DROPPED (a truncated completion is invalid JSON
    and would poison the signal). Returns (kept_examples, n_dropped).
    """
    eos = tokenizer.eos_token_id
    kept: list[dict] = []
    dropped = 0
    for r in rows:
        prompt_text = tokenizer.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(r["completion"], add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + completion_ids + [eos]
        if len(input_ids) > max_seq_len:
            dropped += 1
            continue
        labels = ([-100] * len(prompt_ids)) + completion_ids + [eos]
        kept.append({"input_ids": input_ids, "labels": labels, "length": len(input_ids)})
    return kept, dropped


class Collator:
    """Right-pad input_ids / attention_mask; pad labels with -100."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, attn, labels = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
            labels.append(b["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ── Callbacks ───────────────────────────────────────────────────────────────

def _make_callbacks(cfg: Config):
    from transformers import TrainerCallback

    class WallClockCheckpoint(TrainerCallback):
        """Force a save every cfg.checkpoint_minutes of wall-clock time."""

        def __init__(self, minutes: float):
            self.interval = minutes * 60.0
            self.last_save = time.time()

        def on_step_end(self, args, state, control, **kw):
            if time.time() - self.last_save >= self.interval:
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kw):
            self.last_save = time.time()
            return control

    class StatusLine(TrainerCallback):
        """One healthy-at-a-glance line per logging step."""

        def __init__(self, ckpt_interval_s: float):
            self.t0 = time.time()
            self.ckpt_interval_s = ckpt_interval_s
            self.last_save = time.time()

        def on_save(self, args, state, control, **kw):
            self.last_save = time.time()

        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            if "loss" not in logs:
                return
            done, total = state.global_step, state.max_steps
            elapsed = time.time() - self.t0
            it_s = elapsed / max(done, 1)
            eta_ckpt = max(0.0, self.ckpt_interval_s - (time.time() - self.last_save))
            frac = done / max(total, 1)
            eta_done = (elapsed / max(frac, 1e-9)) - elapsed
            print(
                f"step {done:>5}/{total} | ep {logs.get('epoch', 0):4.2f} | "
                f"loss {logs['loss']:.4f} | lr {logs.get('learning_rate', 0):.2e} | "
                f"{it_s:4.1f}s/it | ckpt in {eta_ckpt/60:4.1f}m | "
                f"elapsed {elapsed/60:5.1f}m | eta {eta_done/60:5.1f}m",
                flush=True,
            )

    class StepTimer(TrainerCallback):
        """Per-optimiser-step wall times, for the --calibrate estimate."""

        def __init__(self):
            self.stamps: list[float] = []

        def on_step_end(self, args, state, control, **kw):
            self.stamps.append(time.perf_counter())
            return control

    timer = StepTimer() if cfg.calibrate > 0 else None
    callbacks = [WallClockCheckpoint(cfg.checkpoint_minutes),
                 StatusLine(cfg.checkpoint_minutes * 60.0)]
    if timer is not None:
        callbacks.append(timer)
    return callbacks, timer


def _report_calibration(timer, steps_per_epoch: int, epochs: float,
                        avg_len: int, batch: int, heuristic_note: str) -> dict:
    """Turn measured per-step times into a real full-run estimate. Skips the
    first 5 steps (cudnn autotune / cache warmup inflates them)."""
    stamps = timer.stamps
    if len(stamps) < 8:
        print("!! calibration: too few steps captured for a stable estimate.")
        return {}
    diffs = [b - a for a, b in zip(stamps[5:-1], stamps[6:])]
    diffs.sort()
    s_per_step = diffs[len(diffs) // 2]           # median
    full_steps = int(round(steps_per_epoch * epochs))
    proj = s_per_step * full_steps
    tok_s = avg_len * batch / s_per_step
    lo, hi = proj * 1.05, proj * 1.12             # + checkpoint/eval overhead band

    def _hms(s: float) -> str:
        h, r = divmod(int(s), 3600)
        m, _ = divmod(r, 60)
        return f"{h}:{m:02d}"

    out = {
        "measured_s_per_step": round(s_per_step, 3),
        "measured_tokens_per_s": round(tok_s, 1),
        "n_steps_measured": len(diffs),
        "projected_full_run_steps": full_steps,
        "projected_full_run_hms": _hms(proj),
        "projected_with_overhead_hms": f"{_hms(lo)}-{_hms(hi)}",
        "heuristic_on_record": heuristic_note,
    }
    print("\n" + "=" * 60)
    print("  CALIBRATION (measured on this T4)")
    print(f"  {out['measured_s_per_step']} s/step  (median of {out['n_steps_measured']} "
          f"steps, first 5 skipped)")
    print(f"  ~{out['measured_tokens_per_s']} tokens/s  (batch {batch} x ~{avg_len} tok)")
    print(f"  full run = {full_steps} steps  ->  {out['projected_full_run_hms']}  "
          f"({out['projected_with_overhead_hms']} with ckpt/eval overhead)")
    print(f"  heuristic on record: {heuristic_note}")
    print("  PASTE BACK to Claude:  "
          f"\"calibration: {out['measured_s_per_step']} s/step, "
          f"projected {out['projected_with_overhead_hms']}\"")
    print("=" * 60, flush=True)
    return out


# ── Resume helper ───────────────────────────────────────────────────────────

def _latest_checkpoint(output_dir: Path):
    if not output_dir.is_dir():
        return None
    cks = [p for p in output_dir.glob("checkpoint-*") if p.is_dir() and (p / "trainer_state.json").exists()]
    if not cks:
        return None
    return str(max(cks, key=lambda p: int(p.name.split("-")[1])))


def _allow_rng_state_unpickling() -> None:
    """
    torch>=2.6 defaults torch.load(weights_only=True); transformers==4.46.3's
    Trainer._load_rng_state calls torch.load(rng_file) with no override, and the
    saved rng_state.pth contains numpy's legacy MT19937 state array, which needs
    these four globals allowlisted to unpickle under the strict default. All
    files this loads are checkpoints this same script just wrote — a trusted
    local source, so allowlisting is safe (found empirically 2026-09-06 by
    replaying an rng_state.pth's unpickling errors one at a time; see
    DECISIONS.md same date for why this is a patch, not a torch/transformers
    version change).
    """
    import numpy as np
    from torch.serialization import add_safe_globals

    add_safe_globals([
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.UInt32DType,
    ])


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    _allow_rng_state_unpickling()
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("!! CUDA not available — this script is meant for a Colab T4. Aborting.")
        sys.exit(1)
    if torch.cuda.is_bf16_supported():
        print("note: bf16 is supported here; the T4 recipe still uses fp16 (see DECISIONS).")

    # ── tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── data ──
    train_path = _DATA_DIR / "train.jsonl"
    val_path = _DATA_DIR / "val.jsonl"
    if not train_path.exists():
        print(f"!! {train_path} missing — run `python sft/prepare_sft_data.py` first.")
        sys.exit(1)

    t_rows = _load_pairs(train_path, limit=cfg.limit)
    v_rows = _load_pairs(val_path, limit=cfg.eval_subset)
    print(f"tokenising {len(t_rows)} train / {len(v_rows)} eval pairs (max_seq_len={cfg.max_seq_len}) ...")
    train_ds, n_drop = _tokenize_pairs(t_rows, tokenizer, cfg.max_seq_len)
    eval_ds, _ = _tokenize_pairs(v_rows, tokenizer, cfg.max_seq_len)
    print(f"  kept {len(train_ds)} train  (dropped {n_drop} over-length, "
          f"{100*n_drop/max(len(t_rows),1):.1f}%)  |  {len(eval_ds)} eval")

    lengths = sorted(e["length"] for e in train_ds)
    avg_len = sum(lengths) // max(len(lengths), 1)
    print(f"  token lengths: p50={lengths[len(lengths)//2]}  "
          f"p90={lengths[int(len(lengths)*0.9)]}  max={lengths[-1]}  mean={avg_len}")

    # HF Dataset so group_by_length can read the "length" column directly
    # (a plain list of dicts breaks LengthGroupedSampler's column access).
    train_ds = Dataset.from_list(train_ds)
    eval_ds = Dataset.from_list(eval_ds)

    steps_per_epoch = math.ceil(len(train_ds) / (cfg.per_device_batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.num_train_epochs if cfg.max_steps < 0 else cfg.max_steps
    print(f"  ~{steps_per_epoch} steps/epoch  ->  ~{int(total_steps)} total optimiser steps")

    calibrating = cfg.calibrate > 0
    if calibrating:
        print(f"  CALIBRATION MODE: {cfg.calibrate} steps, no checkpoint/eval, then exit.")

    # ── model (4-bit NF4, fp16 compute — T4 has no bf16) ──
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",   # no flash-attn on Turing
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.calibrate if calibrating else cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        optim=cfg.optim,
        fp16=True,
        bf16=False,
        logging_steps=cfg.logging_steps,
        eval_strategy="no" if calibrating else "steps",
        eval_steps=cfg.eval_steps,
        save_strategy="no" if calibrating else "steps",
        save_steps=cfg.save_steps_floor,
        save_total_limit=cfg.save_total_limit,
        group_by_length=True,
        report_to="none",
        seed=cfg.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        disable_tqdm=True,
    )

    callbacks, timer = _make_callbacks(cfg)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=None if calibrating else eval_ds,
        data_collator=Collator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )

    resume = None if calibrating else _latest_checkpoint(out_dir)
    (out_dir / "run_config.json").write_text(
        json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in vars(cfg).items()}, indent=2)
    )
    print("=" * 78)
    print(f"{'RESUMING from ' + resume if resume else 'STARTING fresh'}")
    print(f"output_dir   {out_dir}")
    print(f"checkpoint   every {cfg.checkpoint_minutes} min (+ every {cfg.save_steps_floor} steps)")
    print("=" * 78, flush=True)

    trainer.train(resume_from_checkpoint=resume)

    if calibrating:
        report = _report_calibration(
            timer, steps_per_epoch, cfg.num_train_epochs, avg_len,
            cfg.per_device_batch_size,
            heuristic_note="~1 hr for 3 epochs (DECISIONS.md 2026-08-31, pre-calibration)",
        )
        if report:
            (out_dir / "calibration.json").write_text(json.dumps(report, indent=2))
        print("calibration done — no checkpoint written. Report the number, then "
              "run without --calibrate for the real thing.")
        return

    final = out_dir / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    (out_dir / "TRAINING_COMPLETE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"\ndone. adapter + tokenizer -> {final}")


def _parse_args() -> Config:
    d = Config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default=d.model_name)
    ap.add_argument("--output-dir", default=d.output_dir)
    ap.add_argument("--max-seq-len", type=int, default=d.max_seq_len)
    ap.add_argument("--lora-r", type=int, default=d.lora_r)
    ap.add_argument("--lora-alpha", type=int, default=d.lora_alpha)
    ap.add_argument("--lora-dropout", type=float, default=d.lora_dropout)
    ap.add_argument("--learning-rate", type=float, default=d.learning_rate)
    ap.add_argument("--num-train-epochs", type=float, default=d.num_train_epochs)
    ap.add_argument("--per-device-batch-size", type=int, default=d.per_device_batch_size)
    ap.add_argument("--grad-accum", type=int, default=d.grad_accum)
    ap.add_argument("--checkpoint-minutes", type=float, default=d.checkpoint_minutes)
    ap.add_argument("--save-steps-floor", type=int, default=d.save_steps_floor)
    ap.add_argument("--eval-steps", type=int, default=d.eval_steps)
    ap.add_argument("--eval-subset", type=int, default=d.eval_subset)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--limit", type=int, default=d.limit, help="use only first N train pairs (dev)")
    ap.add_argument("--max-steps", type=int, default=d.max_steps, help="hard cap on steps (dev)")
    ap.add_argument("--calibrate", type=int, default=d.calibrate,
                    help="run N steps, print a measured full-run time estimate, then exit "
                         "(no checkpoint). Use ~50.")
    a = ap.parse_args()
    return Config(
        model_name=a.model_name, output_dir=a.output_dir, max_seq_len=a.max_seq_len,
        lora_r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
        learning_rate=a.learning_rate, num_train_epochs=a.num_train_epochs,
        per_device_batch_size=a.per_device_batch_size, grad_accum=a.grad_accum,
        checkpoint_minutes=a.checkpoint_minutes, save_steps_floor=a.save_steps_floor,
        eval_steps=a.eval_steps, eval_subset=a.eval_subset, seed=a.seed,
        limit=a.limit, max_steps=a.max_steps, calibrate=a.calibrate,
    )


if __name__ == "__main__":
    main()

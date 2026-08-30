"""
End-to-end ResPlan -> project-schema conversion driver (Day 3 phase B).

    python -m data.run                     # full run; writes data/processed/*
    python -m data.run --limit 500         # quick smoke run on the first 500 plans
    python -m data.run --emit-samples      # (re)write data/samples/*.json

Bulk outputs land in data/processed/ (git-ignored). A small, curated set of
converted plans is written to data/samples/ (committed) for reference and tests.

Checkpoint/resume: conversion results are cached to data/processed/_convert_cache.pkl
every _CKPT_EVERY plans and on interrupt; a re-run without --no-resume reuses it.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

from shapely.geometry import Polygon

from data.convert import _largest_inner_ring, _polys, convert_plan
from data.dedup_split import Entry, cluster_entries, inner_hash, normalized_union, split_clusters
from verifiers import egress, geometry_gate, topology

_ROOT = Path(__file__).resolve().parent.parent
_RAW = _ROOT / "data" / "raw" / "ResPlan.pkl"
_PROCESSED = _ROOT / "data" / "processed"
_SAMPLES = _ROOT / "data" / "samples"
_CACHE = _PROCESSED / "_convert_cache.pkl"
_CKPT_EVERY = 2000


# ── Conversion pass ─────────────────────────────────────────────────────────

def _convert_all(raw_plans: list[dict], limit: int | None, resume: bool) -> list[dict]:
    """
    Convert every raw plan. Returns a list of result records:
      {"plan_id", "status": "ok"|"dropped", "reason", "plan_json"|None,
       "gate_pass": bool|None, "gate_codes": [str]}
    """
    cache: dict[int, dict] = {}
    if resume and _CACHE.exists():
        cache = pickle.loads(_CACHE.read_bytes())
        print(f"  resumed: {len(cache)} cached conversion results")

    plans = raw_plans[:limit] if limit else raw_plans
    results: list[dict] = []
    t0 = time.time()
    for i, rp in enumerate(plans):
        pid = rp.get("id", i)
        if pid in cache:
            results.append(cache[pid])
            continue

        plan_json, reason = convert_plan(rp)
        if plan_json is None:
            rec = {"plan_id": pid, "status": "dropped", "reason": reason,
                   "plan_json": None, "gate_pass": None, "gate_codes": []}
        else:
            gate = geometry_gate.check(plan_json)
            codes = sorted({iss["code"] for iss in gate["issues"]})
            if gate["pass"]:
                rec = {"plan_id": pid, "status": "ok", "reason": None,
                       "plan_json": plan_json, "gate_pass": True, "gate_codes": []}
            else:
                rec = {"plan_id": pid, "status": "dropped",
                       "reason": "geometry_gate_fail", "plan_json": plan_json,
                       "gate_pass": False, "gate_codes": codes}
        cache[pid] = rec
        results.append(rec)

        if (i + 1) % _CKPT_EVERY == 0:
            _PROCESSED.mkdir(parents=True, exist_ok=True)
            _CACHE.write_bytes(pickle.dumps(cache))
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(plans)}  ({rate:.0f} plans/s)")

    _PROCESSED.mkdir(parents=True, exist_ok=True)
    _CACHE.write_bytes(pickle.dumps(cache))
    return results


# ── Build dedup/split entries ──────────────────────────────────────────────

def _build_entries(raw_by_id: dict[int, dict], ok_records: list[dict]) -> list[Entry]:
    entries: list[Entry] = []
    for rec in ok_records:
        pid = rec["plan_id"]
        rp = raw_by_id[pid]
        pj = rec["plan_json"]
        inner = _largest_inner_ring(rp)
        room_polys = [Polygon(r["polygon"]) for r in pj["rooms"]]
        nunion = normalized_union(room_polys)
        bx0, by0, bx1, by1 = Polygon(pj["boundary"]["polygon"]).bounds
        entries.append(Entry(
            plan_id=pid,
            inner_hash=inner_hash(inner),
            area_m2=float(rp.get("area") or 0.0),
            bbox_w=bx1 - bx0,
            bbox_h=by1 - by0,
            room_count=len(pj["rooms"]),
            bedroom_count=len(_polys(rp.get("bedroom"))),
            norm_union=nunion,
        ))
    return entries


# ── Sample selection ──────────────────────────────────────────────────────

def _select_samples(raw_by_id: dict[int, dict], ok_records: list[dict]) -> list[tuple[str, dict]]:
    """Pick a spread of ~9 converted plans for data/samples/. Returns (filename, plan_json)."""
    from data.convert import _polys  # noqa: PLC0415

    picked: list[tuple[str, dict]] = []
    seen: set[int] = set()

    def take(tag: str, rec: dict) -> None:
        if rec["plan_id"] in seen:
            return
        seen.add(rec["plan_id"])
        picked.append((f"sample_{tag}_id{rec['plan_id']}.json", rec["plan_json"]))

    def max_verts(pj: dict) -> int:
        return max((len(r["polygon"]) for r in pj["rooms"]), default=0)

    def promoted_window(pj: dict) -> bool:  # interior opening ResPlan traced as a window
        return any(o["id"].startswith("window_") and o["type"] == "door" for o in pj["openings"])

    def exterior_window(pj: dict) -> bool:
        return any(o["type"] == "window" for o in pj["openings"])

    def raw_had_overlap(pid: int) -> bool:
        polys = [g for k in ("living", "bedroom", "bathroom", "kitchen", "storage", "balcony")
                 for g in _polys(raw_by_id[pid].get(k))]
        return any(polys[i].intersection(polys[j]).area > 1e-6
                   for i in range(len(polys)) for j in range(i + 1, len(polys)))

    def first(pred):
        return next((r for r in ok_records if r["plan_id"] not in seen and pred(r)), None)

    wanted = [
        ("roomcount_5_min", lambda r: r["plan_json"]["metadata"]["room_count_in_scope"] == 5),
        ("roomcount_8_max", lambda r: r["plan_json"]["metadata"]["room_count_in_scope"] == 8),
        ("open_plan_promoted_window", lambda r: promoted_window(r["plan_json"])),
        ("exterior_window", lambda r: exterior_window(r["plan_json"]) and not promoted_window(r["plan_json"])),
        ("overlap_fixed", lambda r: raw_had_overlap(r["plan_id"])),
    ]
    for tag, pred in wanted:
        rec = first(pred)
        if rec:
            take(tag, rec)

    jag = max(ok_records, key=lambda r: max_verts(r["plan_json"]), default=None)
    if jag:
        take("jagged_polygon", jag)

    def fully_clean(pj: dict) -> bool:  # geometry_gate already OK; also topo + egress
        if not topology.check(pj)["pass"]:
            return False
        return all(r["status"] == "PASS" for r in egress.check(pj))

    for rec in ok_records:  # fill out with clean, unremarkable exemplars
        if len(picked) >= 9:
            break
        pj = rec["plan_json"]
        if max_verts(pj) <= 16 and fully_clean(pj):
            take("clean_exemplar", rec)
    return picked[:9]


# ── Output ────────────────────────────────────────────────────────────────

def _write_splits(entries: list[Entry], records_by_id: dict[int, dict]) -> None:
    _PROCESSED.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for e in entries:
        pj = dict(records_by_id[e.plan_id]["plan_json"])
        pj["metadata"] = {**pj["metadata"], "cluster_id": e.cluster_id, "split": e.split}
        buckets[e.split].append(pj)
    for split, plans in buckets.items():
        path = _PROCESSED / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for pj in plans:
                fh.write(json.dumps(pj) + "\n")
        print(f"  wrote {path.relative_to(_ROOT)}  ({len(plans)} plans)")


def _write_manifest(results: list[dict], entries: list[Entry]) -> None:
    split_by_id = {e.plan_id: (e.split, e.cluster_id) for e in entries}
    rows = []
    for rec in results:
        split, cluster = split_by_id.get(rec["plan_id"], (None, None))
        rows.append({
            "plan_id": rec["plan_id"],
            "status": rec["status"],
            "reason": rec["reason"],
            "gate_pass": rec["gate_pass"],
            "gate_codes": rec["gate_codes"],
            "split": split,
            "cluster_id": cluster,
        })
    path = _PROCESSED / "conversion_manifest.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"  wrote {path.relative_to(_ROOT)}")


def _write_samples(samples: list[tuple[str, dict]]) -> None:
    _SAMPLES.mkdir(parents=True, exist_ok=True)
    for name, pj in samples:
        (_SAMPLES / name).write_text(json.dumps(pj, indent=2), encoding="utf-8")
    print(f"  wrote {len(samples)} sample(s) to {_SAMPLES.relative_to(_ROOT)}/")


# ── Reporting ─────────────────────────────────────────────────────────────

def _report(results: list[dict], entries: list[Entry], split_counts: dict[str, int],
            clusters: list[list[Entry]]) -> None:
    n = len(results)
    reasons = Counter(r["reason"] for r in results if r["status"] == "dropped")
    gate_codes = Counter(c for r in results for c in r["gate_codes"])
    n_ok = sum(1 for r in results if r["status"] == "ok")

    print("\n" + "=" * 64)
    print("CONVERSION SUMMARY")
    print("=" * 64)
    print(f"  raw plans processed:         {n}")
    print(f"  converted + geometry_gate OK: {n_ok}")
    print("  dropped, by reason:")
    for reason, c in reasons.most_common():
        print(f"    {reason:28s} {c}")
    if gate_codes:
        print("  geometry_gate failure codes (within geometry_gate_fail drops):")
        for code, c in gate_codes.most_common():
            print(f"    {code:28s} {c}")
    print(f"\n  clusters: {len(clusters)}  (largest {max((len(c) for c in clusters), default=0)} plans)")
    multi = sum(1 for c in clusters if len(c) > 1)
    print(f"  multi-plan clusters (near-dupes/augmented): {multi}")
    print(f"  final split (by cluster, 80/10/10 by plan count):")
    tot = sum(split_counts.values())
    for s in ("train", "val", "test"):
        pct = split_counts[s] / tot * 100 if tot else 0
        print(f"    {s:5s} {split_counts[s]:6d}  ({pct:.1f}%)")
    print(f"    total {tot}")


# ── Main ──────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N raw plans")
    ap.add_argument("--no-resume", action="store_true", help="ignore the conversion cache")
    ap.add_argument("--emit-samples", action="store_true", help="also (re)write data/samples/*.json")
    ap.add_argument("--samples-only", action="store_true",
                    help="only (re)write samples from the cache, skip split/manifest writes")
    args = ap.parse_args(argv)

    if not _RAW.exists():
        print(f"ERROR: {_RAW} not found - see data/raw/PROVENANCE.md for download steps.",
              file=sys.stderr)
        return 1

    print(f"loading {_RAW.relative_to(_ROOT)} ...")
    raw_plans: list[dict] = pickle.loads(_RAW.read_bytes())
    raw_by_id = {rp.get("id", i): rp for i, rp in enumerate(raw_plans)}
    print(f"  {len(raw_plans)} raw plans")

    print("converting ...")
    results = _convert_all(raw_plans, args.limit, resume=not args.no_resume)
    ok_records = [r for r in results if r["status"] == "ok"]
    records_by_id = {r["plan_id"]: r for r in ok_records}

    print("clustering + splitting ...")
    entries = _build_entries(raw_by_id, ok_records)
    clusters = cluster_entries(entries)
    split_counts = split_clusters(clusters)

    if not args.samples_only:
        _write_splits(entries, records_by_id)
        _write_manifest(results, entries)

    if args.emit_samples or args.samples_only:
        _write_samples(_select_samples(raw_by_id, ok_records))

    _report(results, entries, split_counts, clusters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

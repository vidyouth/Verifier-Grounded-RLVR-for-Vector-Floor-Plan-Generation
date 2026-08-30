# ResPlan raw data — provenance & download

## Source
- Repository: https://github.com/m-agour/ResPlan  (commit cloned 2026-08-30)
- Kaggle mirror (geometry + prebuilt `plan["graph"]`): https://www.kaggle.com/datasets/resplan/resplan
- Paper: "ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor
  Plans" (under peer review as of 2026-08-30; citation withheld upstream).

## How to obtain (not committed to git)
`ResPlan.pkl` and `ResPlan.zip` are git-ignored. To reproduce this directory:

```bash
git clone --depth 1 https://github.com/m-agour/ResPlan.git /tmp/ResPlan
cp /tmp/ResPlan/ResPlan.zip  data/raw/
cd data/raw && unzip -o ResPlan.zip        # -> ResPlan.pkl  (~247 MB)
```

`split.json`, `croissant.json`, `resplan_utils.py`, `README.upstream.md`,
`LICENSE.upstream`, `TAKEDOWN.md` are small and kept in git for reproducibility.

## Licensing
- **Data** (`ResPlan.pkl`, `split.json`, `croissant.json`): Creative Commons
  Attribution 4.0 International (CC BY 4.0) — https://creativecommons.org/licenses/by/4.0/
- **Code** (`resplan_utils.py`, upstream notebook, baselines): MIT,
  Copyright (c) 2025 The ResPlan Authors.
- Full upstream text: `LICENSE.upstream`.

### Attribution obligation (CC BY 4.0) — TODO, not yet added
A CC BY 4.0 attribution notice for ResPlan must appear in:
1. Top-level `README.md` of this repo (a "Data" section) — **not yet written**.
2. The final paper / write-up, in the dataset section and references.
3. `docs/LIMITATIONS.md` cross-reference (data-source-differs entry — added 2026-08-30).
Attribution should credit "The ResPlan Authors", link the repo and the CC BY 4.0
deed, and state that the data was used unmodified as input to a schema-conversion
pipeline. If any derived/normalized dataset is redistributed, that derivative must
also carry CC BY 4.0 + a note of changes.

## Takedown
Upstream `TAKEDOWN.md` applies. If upstream publishes `REMOVED.md`, drop those
plan IDs from our splits before any release.

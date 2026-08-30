"""
Geometric dedup + cluster-level train/val/test split (Day 3 phase B).

ResPlan's canonical `split.json` is deliberately NOT used: it has known
train/test leakage (154 near-duplicate test plans) and no building-identity
field to group by. Instead:

  1. Augmented plans are linked to their origin by an exact hash of the raw
     `inner` polygon vertices rounded to 0.1 canvas units (matched 667 of 683
     pairs in phase A; rotated augmentations fall back to the IoU test below).
  2. Original near-duplicates are found by bucketing on
     (area 0.5 m², bbox W/H 0.25 m, room count, bedroom count) and confirming a
     pair as duplicate when IoU of the origin-aligned room-union ≥ 0.90.
  3. Union-Find over both link types yields clusters. Whole clusters are assigned
     to train/val/test 80/10/10 by plan count, so no cluster spans two splits.

See docs/DECISIONS.md 2026-08-30.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

_IOU_DUP_THRESHOLD: float = 0.90
_SPLIT_RATIOS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


def inner_hash(inner_polygon: Polygon) -> str:
    """Exact hash of an `inner` ring's vertices rounded to 0.1 canvas units."""
    pts = tuple(sorted((round(x, 1), round(y, 1)) for x, y in inner_polygon.exterior.coords))
    return hashlib.md5(str(pts).encode()).hexdigest()[:16]


@dataclass
class Entry:
    """One converted plan, reduced to what dedup/split needs."""
    plan_id: int
    inner_hash: str
    area_m2: float
    bbox_w: float
    bbox_h: float
    room_count: int
    bedroom_count: int
    norm_union: Polygon | MultiPolygon  # room-union translated so bbox min = (0, 0)
    split: str | None = field(default=None)
    cluster_id: int | None = field(default=None)

    @property
    def bucket(self) -> tuple:
        return (
            round(self.area_m2 * 2) / 2,       # 0.5 m²
            round(self.bbox_w * 4) / 4,        # 0.25 m
            round(self.bbox_h * 4) / 4,        # 0.25 m
            self.room_count,
            self.bedroom_count,
        )


def normalized_union(room_polys: Iterable[Polygon]) -> Polygon | MultiPolygon:
    """Union of room polygons translated so the union's bbox min is at the origin."""
    u = unary_union(list(room_polys))
    minx, miny, _, _ = u.bounds
    from shapely.affinity import translate
    return translate(u, xoff=-minx, yoff=-miny)


def _iou(a: Polygon | MultiPolygon, b: Polygon | MultiPolygon) -> float:
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def cluster_entries(entries: list[Entry]) -> list[list[Entry]]:
    """Populate entry.cluster_id and return clusters as lists of entries."""
    uf = _UnionFind(len(entries))

    # link 1: identical raw inner hash (augmented ↔ origin, exact republished dupes)
    by_hash: dict[str, list[int]] = {}
    for idx, e in enumerate(entries):
        by_hash.setdefault(e.inner_hash, []).append(idx)
    for idxs in by_hash.values():
        for k in idxs[1:]:
            uf.union(idxs[0], k)

    # link 2: same coarse bucket AND origin-aligned room-union IoU ≥ threshold
    by_bucket: dict[tuple, list[int]] = {}
    for idx, e in enumerate(entries):
        by_bucket.setdefault(e.bucket, []).append(idx)
    for idxs in by_bucket.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                if uf.find(ia) == uf.find(ib):
                    continue
                if _iou(entries[ia].norm_union, entries[ib].norm_union) >= _IOU_DUP_THRESHOLD:
                    uf.union(ia, ib)

    groups: dict[int, list[Entry]] = {}
    for idx, e in enumerate(entries):
        root = uf.find(idx)
        e.cluster_id = root
        groups.setdefault(root, []).append(e)

    # renumber cluster ids 0..k-1 deterministically (by smallest plan_id in cluster)
    ordered = sorted(groups.values(), key=lambda g: min(x.plan_id for x in g))
    for new_id, grp in enumerate(ordered):
        for e in grp:
            e.cluster_id = new_id
    return ordered


def split_clusters(clusters: list[list[Entry]]) -> dict[str, int]:
    """
    Greedily assign whole clusters to train/val/test to hit 80/10/10 by plan
    count. Larger clusters first; each goes to whichever split is furthest below
    its target. Mutates entry.split. Returns per-split plan counts.
    """
    total = sum(len(c) for c in clusters)
    counts = {s: 0 for s in _SPLIT_RATIOS}
    order = sorted(clusters, key=lambda c: (-len(c), min(e.plan_id for e in c)))
    for grp in order:
        deficit = {s: _SPLIT_RATIOS[s] * total - counts[s] for s in _SPLIT_RATIOS}
        target = max(deficit, key=lambda s: (deficit[s], -counts[s]))
        for e in grp:
            e.split = target
        counts[target] += len(grp)
    return counts

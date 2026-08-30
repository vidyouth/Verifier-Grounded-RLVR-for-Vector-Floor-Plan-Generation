"""Tests for data/dedup_split.py — geometric clustering + cluster-level split."""

from __future__ import annotations

from shapely.affinity import translate
from shapely.geometry import box

from data.dedup_split import (
    Entry,
    cluster_entries,
    inner_hash,
    normalized_union,
    split_clusters,
)


def _entry(pid: int, ih: str, union, area=100.0, w=10.0, h=10.0, rc=6, bc=2) -> Entry:
    return Entry(plan_id=pid, inner_hash=ih, area_m2=area, bbox_w=w, bbox_h=h,
                 room_count=rc, bedroom_count=bc, norm_union=union)


def _u(*boxes):
    return normalized_union([box(*b) for b in boxes])


# ── inner_hash ─────────────────────────────────────────────────────────────

def test_inner_hash_stable_and_discriminating():
    a = box(0, 0, 10, 8)
    assert inner_hash(a) == inner_hash(box(0, 0, 10, 8))
    assert inner_hash(a) != inner_hash(box(0, 0, 10, 9))


def test_inner_hash_translation_sensitive():
    # raw inner polygons are in absolute canvas coords; a shifted plan is a
    # different raw hash (augmented rotations are caught by the IoU link instead).
    a = box(0, 0, 10, 8)
    assert inner_hash(a) != inner_hash(translate(a, xoff=5))


# ── clustering ────────────────────────────────────────────────────────────

def test_identical_inner_hash_links_regardless_of_geometry_bucket():
    e1 = _entry(1, "HASH_A", _u((0, 0, 4, 4)), area=50, w=4, h=4)
    e2 = _entry(2, "HASH_A", _u((0, 0, 9, 9)), area=200, w=9, h=9)  # different bucket
    clusters = cluster_entries([e1, e2])
    assert len(clusters) == 1
    assert {e1.cluster_id, e2.cluster_id} == {0}


def test_near_duplicate_geometry_links_via_iou():
    base = [(0, 0, 5, 3), (5, 0, 8, 3), (0, 3, 8, 6)]
    e1 = _entry(1, "h1", normalized_union([box(*b) for b in base]))
    jittered = [(0, 0, 5.02, 3), (5.02, 0, 8, 3), (0, 3, 8, 6.01)]
    e2 = _entry(2, "h2", normalized_union([box(*b) for b in jittered]))
    e3 = _entry(3, "h3", _u((0, 0, 8, 3)))  # genuinely different layout, same bucket-ish
    clusters = cluster_entries([e1, e2, e3])
    ids = {e.plan_id: e.cluster_id for e in (e1, e2, e3)}
    assert ids[1] == ids[2]
    assert ids[3] != ids[1]


def test_distinct_plans_stay_separate():
    e1 = _entry(1, "h1", _u((0, 0, 5, 5)), area=25, w=5, h=5)
    e2 = _entry(2, "h2", _u((0, 0, 9, 4)), area=36, w=9, h=4)
    clusters = cluster_entries([e1, e2])
    assert len(clusters) == 2


def test_cluster_ids_are_deterministic_by_min_plan_id():
    # distinct hashes AND distinct geometry/bucket -> 3 singleton clusters,
    # renumbered by smallest plan_id in the cluster
    es = [
        _entry(7, "h7", _u((0, 0, 5, 5)), area=25, w=5, h=5),
        _entry(3, "h3", _u((0, 0, 9, 4)), area=36, w=9, h=4),
        _entry(9, "h9", _u((0, 0, 6, 7)), area=42, w=6, h=7),
    ]
    clusters = cluster_entries(es)
    assert [min(e.plan_id for e in c) for c in clusters] == [3, 7, 9]
    assert [c[0].cluster_id for c in clusters] == [0, 1, 2]


# ── split ────────────────────────────────────────────────────────────────

def _make_clusters(sizes: list[int]) -> list[list[Entry]]:
    out, pid = [], 0
    for cid, sz in enumerate(sizes):
        grp = [_entry(pid + i, f"h{pid + i}", _u((0, 0, 5, 5))) for i in range(sz)]
        for e in grp:
            e.cluster_id = cid
        out.append(grp)
        pid += sz
    return out


def test_split_keeps_clusters_whole():
    clusters = _make_clusters([5, 4, 3, 3, 2, 2, 2, 1, 1, 1] * 6)
    counts = split_clusters(clusters)
    for grp in clusters:
        assert len({e.split for e in grp}) == 1, "a cluster was split across sets"
    assert set(counts) == {"train", "val", "test"}
    assert sum(counts.values()) == sum(len(c) for c in clusters)


def test_split_ratios_approximately_80_10_10():
    clusters = _make_clusters([1] * 500 + [2] * 40 + [5] * 8)
    counts = split_clusters(clusters)
    total = sum(counts.values())
    assert counts["train"] / total == __import__("pytest").approx(0.8, abs=0.05)
    assert counts["val"] / total == __import__("pytest").approx(0.1, abs=0.05)
    assert counts["test"] / total == __import__("pytest").approx(0.1, abs=0.05)


def test_split_is_deterministic():
    c1 = _make_clusters([3, 2, 2, 1, 1, 1, 1] * 10)
    c2 = _make_clusters([3, 2, 2, 1, 1, 1, 1] * 10)
    assert split_clusters(c1) == split_clusters(c2)
    assert [e.split for g in c1 for e in g] == [e.split for g in c2 for e in g]

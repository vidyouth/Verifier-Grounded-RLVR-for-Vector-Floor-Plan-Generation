"""
Tests for the ResPlan -> project-schema conversion pipeline (data/convert.py).

Fixtures are synthetic ResPlan-style plan dicts (Shapely geometry under the raw
ResPlan keys), so the suite is hermetic — it does not need the 247 MB pickle.
`_grid_plan(n)` tiles n rooms on a gapped grid with a connected door chain and a
front door, at unit scale (1 canvas unit == 1 m).
"""

from __future__ import annotations

import copy
import glob
import json
import math
import os

import pytest
from shapely.affinity import scale as shp_scale
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from data.convert import convert_plan, exclusion_reason
from verifiers import corridor, geometry_gate, topology
from rlvr.reward import compute_reward

_CELL = 3.0
_GAP = 0.2
_PITCH = _CELL + _GAP
_TYPES = ["living", "bedroom", "bathroom", "kitchen", "storage"]


def _grid_plan(n: int, sf: float = 1.0, plan_id: int = 900000) -> dict:
    """A synthetic raw-ResPlan plan: n rooms on a `cols` x `rows` gapped grid,
    a door in every wall gap along the room chain, plus a front door on room 0."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    def cell(k: int) -> tuple[int, int]:
        return k % cols, k // cols

    rooms_by_type: dict[str, list[Polygon]] = {t: [] for t in _TYPES}
    for k in range(n):
        c, r = cell(k)
        x0, y0 = c * _PITCH, r * _PITCH
        rooms_by_type[_TYPES[k % len(_TYPES)]].append(box(x0, y0, x0 + _CELL, y0 + _CELL))

    doors: list[Polygon] = []
    for k in range(1, n):
        c, r = cell(k)
        if c != 0:  # horizontal gap to the left neighbour
            xg0, xg1 = c * _PITCH - _GAP, c * _PITCH
            yc = r * _PITCH + _CELL / 2
            doors.append(box(xg0, yc - 0.45, xg1, yc + 0.45))
        elif r != 0:  # first room of a row -> vertical gap to the row above
            yg0, yg1 = r * _PITCH - _GAP, r * _PITCH
            doors.append(box(1.05, yg0, 1.95, yg1))

    front_door = box(-0.1, 1.05, 0.1, 1.95)  # on room 0's outer (x=0) edge

    max_x = (cols - 1) * _PITCH + _CELL
    max_y = (rows - 1) * _PITCH + _CELL
    inner = box(0.0, 0.0, max_x, max_y)
    area_m2 = inner.area  # scale == 1 at sf == 1

    plan: dict = {
        "id": plan_id,
        "area": area_m2,
        "net_area": 0,
        "wall_depth": _GAP,
        "inner": MultiPolygon([inner]),
        "land": MultiPolygon([box(-1, -1, max_x + 1, max_y + 1)]),
        "neighbor": (MultiPolygon([]), MultiPolygon([])),
        "front_door": front_door,
        "door": MultiPolygon(doors),
        "window": None,
        "garden": MultiPolygon([]),
        "parking": MultiPolygon([]),
        "pool": MultiPolygon([]),
    }
    for t, polys in rooms_by_type.items():
        if polys:
            plan[t] = polys[0] if len(polys) == 1 else MultiPolygon(polys)

    if sf != 1.0:
        for key in list(plan):
            v = plan[key]
            if isinstance(v, (Polygon, MultiPolygon)) and not v.is_empty:
                plan[key] = shp_scale(v, xfact=sf, yfact=sf, origin=(0, 0))
        plan["area"] = area_m2  # unchanged -> per-plan scale becomes sf

    return plan


def _endpoint_on_host_wall(pj: dict, tol: float = 1e-6) -> bool:
    rpoly = {r["id"]: Polygon(r["polygon"]) for r in pj["rooms"]}
    bpoly = Polygon(pj["boundary"]["polygon"])
    for o in pj["openings"]:
        edges: list[LineString] = []
        for hid in o["host_wall"]:
            p = bpoly if hid == "exterior" else rpoly.get(hid)
            if p is None:
                continue
            cc = list(p.exterior.coords)
            edges += [LineString([cc[i], cc[i + 1]]) for i in range(len(cc) - 1)]
        for pt in o["endpoints"]:
            if not edges or min(Point(pt).distance(e) for e in edges) > tol:
                return False
    return True


# ── Baseline conversion ─────────────────────────────────────────────────────

def test_baseline_converts_cleanly():
    pj, reason = convert_plan(_grid_plan(6))
    assert reason is None and pj is not None
    assert set(pj) == {"metadata", "boundary", "rooms", "openings", "relations", "audit"}
    assert pj["audit"] == []
    assert "derived" in pj["relations"]
    assert pj["metadata"]["room_count_in_scope"] == 6
    assert len(pj["boundary"]["entrance"]) == 2


def test_baseline_passes_geometry_gate():
    pj, _ = convert_plan(_grid_plan(6))
    result = geometry_gate.check(pj)
    assert result["pass"], result["issues"]


def test_baseline_endpoints_lie_on_host_walls():
    pj, _ = convert_plan(_grid_plan(7))
    assert _endpoint_on_host_wall(pj)


def test_baseline_topology_connected_and_not_hard_gated():
    pj, _ = convert_plan(_grid_plan(6))
    assert topology.check(pj)["pass"]
    r = compute_reward(pj)
    assert r["gate_reason"] is None
    assert r["reward"] > -1.0


def test_convert_plan_does_not_mutate_input():
    raw = _grid_plan(6)
    before = copy.deepcopy(raw)
    convert_plan(raw)
    assert raw.keys() == before.keys()
    assert raw["area"] == before["area"]
    assert raw["inner"].equals(before["inner"])


# ── Exclusion filters ───────────────────────────────────────────────────────

def test_stair_plan_excluded():
    raw = _grid_plan(6)
    raw["stair"] = box(0.2, 0.2, 0.8, 0.8)  # any stair polygon => multi-floor
    assert exclusion_reason(raw) == "has_stair"
    assert convert_plan(raw) == (None, "has_stair")


@pytest.mark.parametrize("n", [3, 4, 9, 11])
def test_room_count_out_of_scope_excluded(n):
    assert convert_plan(_grid_plan(n)) == (None, "room_count_out_of_scope")


@pytest.mark.parametrize("n", [5, 6, 7, 8])
def test_room_count_in_scope_converts(n):
    pj, reason = convert_plan(_grid_plan(n))
    assert reason is None and pj is not None
    assert pj["metadata"]["room_count_in_scope"] == n
    assert geometry_gate.check(pj)["pass"]


# ── Overlap handling ────────────────────────────────────────────────────────

def test_small_overlap_fixed_smaller_room_yields():
    raw = _grid_plan(5)
    # room 1 (bedroom) shifted left so it eats ~0.15 m^2 of room 0 (living)
    raw["bedroom"] = box(_PITCH - 0.25, 0.0, _PITCH - 0.25 + _CELL, _CELL)
    pj, reason = convert_plan(raw)
    assert reason is None and pj is not None
    result = geometry_gate.check(pj)
    assert result["pass"], result["issues"]
    assert not any(i["code"] == "ROOM_OVERLAP" for i in result["issues"])
    living = Polygon(next(r["polygon"] for r in pj["rooms"] if r["id"] == "living_0"))
    bedroom = Polygon(next(r["polygon"] for r in pj["rooms"] if r["id"] == "bedroom_0"))
    assert living.intersection(bedroom).area < 1e-6


def test_large_overlap_drops_plan():
    raw = _grid_plan(5)
    raw["bedroom"] = box(_PITCH - 1.0, 0.0, _PITCH - 1.0 + _CELL, _CELL)  # ~2.4 m^2 into living
    assert convert_plan(raw) == (None, "room_overlap_large")


# ── Corridor: none synthesised on this dataset ──────────────────────────────

def test_no_room_is_typed_corridor():
    pj, _ = convert_plan(_grid_plan(8))
    assert all(r["type"] != "corridor" for r in pj["rooms"])


def test_corridor_check_returns_no_records():
    pj, _ = convert_plan(_grid_plan(8))
    assert corridor.check(pj) == []


# ── Opening typing ─────────────────────────────────────────────────────────

def test_interior_window_promoted_to_door():
    raw = _grid_plan(5)
    # window in the gap between room 1 (bedroom_0) and room 2 (bathroom_0),
    # offset in y from the existing door there
    raw["window"] = MultiPolygon([box(2 * _PITCH - _GAP, 2.1, 2 * _PITCH, 2.7)])
    pj, _ = convert_plan(raw)
    win = [o for o in pj["openings"] if o["id"].startswith("window_")]
    assert win, "no window-derived opening emitted"
    assert win[0]["type"] == "door"
    assert "exterior" not in win[0]["host_wall"]
    assert {"bedroom_0", "bathroom_0"} == set(win[0]["host_wall"])


def test_exterior_window_stays_window():
    raw = _grid_plan(5)
    raw["window"] = MultiPolygon([box(-0.1, _PITCH + 1.0, 0.1, _PITCH + 1.6)])  # room 3 outer edge
    pj, _ = convert_plan(raw)
    ext = [o for o in pj["openings"] if o["type"] == "window"]
    assert ext and ext[0]["host_wall"][0] == "exterior"


# ── Coordinate normalisation ───────────────────────────────────────────────

@pytest.mark.parametrize("sf", [1.0, 2.5, 0.4])
def test_scale_normalisation_recovers_metric_size(sf):
    pj, reason = convert_plan(_grid_plan(6, sf=sf))
    assert reason is None
    x0, y0, x1, y1 = Polygon(pj["boundary"]["polygon"]).bounds
    cols = math.ceil(math.sqrt(6))
    rows = math.ceil(6 / cols)
    assert (x1 - x0) == pytest.approx((cols - 1) * _PITCH + _CELL, abs=0.02)
    assert (y1 - y0) == pytest.approx((rows - 1) * _PITCH + _CELL, abs=0.02)


# ── Committed sample plans ────────────────────────────────────────────────

_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")
_SAMPLE_FILES = sorted(glob.glob(os.path.join(_SAMPLE_DIR, "*.json")))


@pytest.mark.skipif(not _SAMPLE_FILES, reason="no committed samples yet (run: python -m data.run --emit-samples)")
@pytest.mark.parametrize("path", _SAMPLE_FILES, ids=[os.path.basename(p) for p in _SAMPLE_FILES])
def test_committed_sample_passes_geometry_gate(path):
    with open(path, encoding="utf-8") as fh:
        pj = json.load(fh)
    assert set(pj) >= {"metadata", "boundary", "rooms", "openings", "relations", "audit"}
    result = geometry_gate.check(pj)
    assert result["pass"], f"{os.path.basename(path)}: {result['issues']}"

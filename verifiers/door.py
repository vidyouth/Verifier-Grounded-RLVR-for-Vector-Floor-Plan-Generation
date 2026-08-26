"""
Door clear-width verifier.

PLACEHOLDER THRESHOLD: 0.80 m minimum clear width.
PLACEHOLDER — generic/placeholder value, not a real statutory jurisdiction.
Results must not be read as evidence of real-world building-code compliance.
See LIMITATIONS.md.
"""

from __future__ import annotations

# PLACEHOLDER — generic threshold, not a real jurisdiction (see LIMITATIONS.md)
_MIN_WIDTH_M: float = 0.80
_CLAUSE_ID: str = "DOOR-W-01"
_EPS: float = 1e-9  # margin denominator floor


def check(plan: dict) -> list[dict]:
    """
    Check every opening typed as "door" against the minimum clear-width threshold.

    Returns
    -------
    list of audit records, one per door opening:
        clause_id      : str
        opening_id     : str
        measured_value : float  (metres; absent on ERROR)
        threshold      : float  (metres; absent on ERROR)
        margin         : float  (measured − threshold; negative = fail; absent on ERROR)
        status         : "PASS" | "FAIL" | "ERROR"
        detail         : str    (present only on ERROR or zero/negative clear_width)
    """
    results = []
    for opening in plan.get("openings", []):
        if opening.get("type") != "door":
            continue
        oid = opening["id"]

        if "clear_width" not in opening:
            results.append({
                "clause_id": _CLAUSE_ID,
                "opening_id": oid,
                "status": "ERROR",
                "detail": "missing clear_width field",
            })
            continue

        cw = opening["clear_width"]

        if not isinstance(cw, (int, float)):
            results.append({
                "clause_id": _CLAUSE_ID,
                "opening_id": oid,
                "status": "ERROR",
                "detail": f"clear_width is not numeric: {cw!r}",
            })
            continue

        cw = float(cw)

        if cw <= 0:
            results.append({
                "clause_id": _CLAUSE_ID,
                "opening_id": oid,
                "measured_value": cw,
                "threshold": _MIN_WIDTH_M,
                "margin": round((cw - _MIN_WIDTH_M) / max(abs(_MIN_WIDTH_M), _EPS), 6),
                "status": "FAIL",
                "detail": "zero or negative clear_width",
            })
            continue

        margin = round((cw - _MIN_WIDTH_M) / max(abs(_MIN_WIDTH_M), _EPS), 6)
        results.append({
            "clause_id": _CLAUSE_ID,
            "opening_id": oid,
            "measured_value": cw,
            "threshold": _MIN_WIDTH_M,
            "margin": margin,
            "status": "PASS" if margin >= 0 else "FAIL",
        })

    return results

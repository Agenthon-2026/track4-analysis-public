"""Rule-based reader/reasoner for the minimal Track 4 baseline.

Replaces the open-weights LLM of the full baseline spec with a transparent rule:
for the EPS-beat family it compares a corpus-extracted reported EPS against the
entity's ``consensus_eps`` using the declared threshold; for other families it
falls back to a label from the task's declared vocabulary.

Where a value IS extracted, the point forecast and interval are derived from it. Where none is,
the point forecast is a fallback rather than an estimate, and the interval says so -- see
``_uninformative_band``. This agent demonstrates the submission format; it does not forecast, and
a participant is expected to do materially better than it.
"""

from __future__ import annotations

import math
import re

_EPS_RE = re.compile(r"earnings per share of \$?([0-9]+\.[0-9]+)", re.IGNORECASE)


def extract_eps(text: str) -> float | None:
    m = _EPS_RE.search(text)
    return float(m.group(1)) if m else None


def classify_eps(reported: float, consensus: float, threshold_pct: float) -> str:
    if reported > consensus * (1 + threshold_pct):
        return "beat"
    if reported < consensus * (1 - threshold_pct):
        return "miss"
    return "inline"


#: One order of magnitude beyond anything the entity declares. Chosen as a round, deliberately
#: generous margin, NOT calibrated -- see ``_uninformative_band``.
UNINFORMATIVE_ORDERS = 10.0


def _uninformative_band(entity: dict) -> float:
    """Half-width for a prediction the agent has no evidence for.

    The agent reaches this branch when it extracted nothing from the corpus, so its point
    forecast is a fallback rather than an estimate. Emitting a narrow interval around it asserts
    a confidence it does not have -- measured on the ten public-dev units, that produced 2
    coverage hits in 38 on the five regression units, because the band was the 0.05 floor while
    the targets ranged from hundredths to hundreds.

    The composite scores an interval as ``|coverage - level|`` with NO width penalty, so a narrow
    interval buys nothing and costs almost the whole calibration weight. Honesty and the score
    agree here: say plainly that the magnitude is unknown.

    Width is one order of magnitude beyond the largest magnitude the ENTITY ITSELF declares, so a
    unit quoted in basis points and one quoted as a ratio get proportionate widths without any
    per-family knowledge. The multiplier is a stated margin, not a fitted constant: nothing here
    is derived from a resolved outcome, which matters because the resolved outcomes of these units
    are deliberately not published (``docs/TRAINING-POLICY.md`` bars calibrating on post-cutoff
    labels, and this agent ships to participants).
    """
    scale = 0.0
    for value in entity.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            scale = max(scale, abs(float(value)))
    return UNINFORMATIVE_ORDERS * max(scale, 1.0)


def predict_entity(
    entity: dict, span_text: str, *, labels: list[str] | None = None
) -> dict:
    """Return {label, point_forecast, lo, hi} for one entity from its top span."""
    consensus = entity.get("consensus_eps")
    threshold = entity.get("threshold_pct", 0.05)
    reported = extract_eps(span_text)
    if consensus is not None and reported is not None:
        label = classify_eps(reported, float(consensus), float(threshold))
        point = reported
        # A symmetric band of +/- one threshold around the extracted value.
        band = max(abs(point) * float(threshold) * 2.0, 0.05)
    else:
        # No usable evidence — predict the neutral class, and do not let the interval
        # pretend the point forecast is an estimate.
        label = "inline"
        point = float(consensus) if consensus is not None else 0.0
        band = _uninformative_band(entity)
    # Labels are a task-owned vocabulary. Preserve the EPS classifier when its
    # label is allowed; otherwise prefer inline if offered, then the first label
    # in the declared order. This is a deterministic format baseline, not a claim
    # that the fallback is the statistically most likely outcome.
    if labels and label not in labels:
        label = "inline" if "inline" in labels else labels[0]
    return {
        "label": label,
        "point_forecast": point,
        "lo": point - band,
        "hi": point + band,
    }

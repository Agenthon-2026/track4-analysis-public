"""Task quantity contracts shared by model output and deterministic fallbacks.

Bounds validate the declared domain, not predictive accuracy or faithfulness.
Fallback intervals are explicit, uncalibrated baseline assumptions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .schema import interval_level, legal_labels, target_name, target_type


def finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str | None
    labels: tuple[str, ...]
    unit: str
    mode: str
    level: float
    lower: float | None
    upper: float | None
    cutoff: str
    resolution: str
    requires_point: bool = True

    @classmethod
    def from_task(
        cls, task: dict[str, Any], entity: dict[str, Any] | None = None
    ) -> TargetSpec:
        target = task.get("target") or {}
        entity = entity or {}
        name = target_name(task).lower()
        prompt = str(task.get("prompt") or "").lower()
        unit = str(
            target.get("unit") or target.get("units") or entity.get("unit") or ""
        ).lower()
        description = " ".join((name, unit))
        probability_prompt = not unit and re.search(
            r"\b(?:predict|estimate|forecast|report|give|provide)\s+"
            r"(?:(?:the|a|an|predicted)\s+)*probability\s+(?:of|that)\b|"
            r"\bpoint[ _-]forecast\s*(?:=|is|as|:)\s*"
            r"(?:(?:your|the|predicted)\s+)*probability\b",
            prompt,
        )
        if "probability" in description or probability_prompt or "credit event" in name:
            mode, unit = "probability", "probability"
        elif "bps" in description and "change" in description:
            mode, unit = "change_bps", "bps"
        elif "growth" in name and ("pct" in name or "percent" in prompt):
            mode, unit = "growth_pct", "percent"
        elif "pct oi" in name or "pct_of_open_interest" in unit:
            mode, unit = "change_pct_oi", "percent of open interest"
        elif "reaction" in name or "return" in name:
            mode, unit = "return_pct", "percent"
        elif "eps" in name:
            mode, unit = "eps", "currency per share"
        elif "bid to cover" in name or "ratio" in description:
            mode, unit = "ratio", "ratio"
        elif (
            "mom" in name
            or "pct" in description
            or "percent" in description
            or unit == "%"
        ):
            mode, unit = "percent", "percent"
        elif "bps" in description or "basis point" in description:
            mode, unit = "level", "bps"
        else:
            mode = "level"
        domain = target.get("domain") if isinstance(target.get("domain"), dict) else {}
        lower = target.get("minimum", domain.get("minimum", domain.get("min")))
        upper = target.get("maximum", domain.get("maximum", domain.get("max")))
        lower = float(lower) if finite_number(lower) else None
        upper = float(upper) if finite_number(upper) else None
        if mode == "probability":
            lower, upper = (
                max(0.0, lower or 0.0),
                min(1.0, upper if upper is not None else 1.0),
            )
        elif mode == "ratio":
            lower = max(0.0, lower or 0.0)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("target domain has reversed bounds")
        requested_point = re.search(
            r"\b(?:give|provide|report|return|submit|include|predict)\s+"
            r"(?:(?:a|the|your|numeric|numerical)\s+)*"
            r"(?:point[ _-]forecast|numeric (?:quantity|value|estimate|target))\b",
            prompt,
        )
        requires_point = (
            target_type(task) != "classification"
            or bool(unit)
            or lower is not None
            or upper is not None
            or bool(requested_point)
        )
        return cls(
            name,
            target_type(task),
            tuple(legal_labels(task)),
            unit,
            mode,
            interval_level(task),
            lower,
            upper,
            str(task.get("cutoff_date") or ""),
            str(
                task.get("resolution_date")
                or entity.get("resolving_release_date")
                or entity.get("expected_report_date")
                or ""
            ),
            requires_point,
        )

    def validate_prediction(self, prediction: dict[str, Any]) -> None:
        if self.kind == "classification" and prediction.get("label") not in self.labels:
            raise ValueError("prediction label is outside the task vocabulary")
        interval = prediction.get("interval")
        if not isinstance(interval, dict):
            raise ValueError("prediction needs a numeric interval")
        point = prediction.get("point_forecast")
        lo, hi = interval.get("lo"), interval.get("hi")
        if not finite_number(lo) or not finite_number(hi):
            raise ValueError("prediction and interval must contain finite numbers")
        if lo > hi:
            raise ValueError("prediction interval must have ordered bounds")
        # The scorer allows an absent numeric point for label-only tasks, but
        # the wire schema still requires an interval. Callers omit null points
        # when serializing, because the schema accepts only numbers if present.
        if point is None and not self.requires_point:
            values = (lo, hi)
        elif not finite_number(point):
            raise ValueError("prediction requires a finite numeric point")
        elif not lo <= point <= hi:
            raise ValueError("prediction must lie inside its ordered interval")
        else:
            values = (point, lo, hi)
        if (
            not finite_number(interval.get("level"))
            or abs(interval["level"] - self.level) > 1e-9
        ):
            raise ValueError("interval level differs from the task")
        if self.lower is not None and min(values) < self.lower:
            raise ValueError("prediction is below the target domain")
        if self.upper is not None and max(values) > self.upper:
            raise ValueError("prediction is above the target domain")

    def interval(self, point: float) -> tuple[float, float]:
        if self.mode == "probability":
            if point > 0.5:
                lo = max(self.lower or 0.0, point - 0.05)
                hi = self.upper if self.upper is not None else 1.0
            else:
                lo = self.lower or 0.0
                hi = min(self.upper if self.upper is not None else 1.0, 0.50)
            return lo, hi
        # Numeric target units also govern intervals on classification tasks.
        if self.mode == "change_bps":
            half = max(25.0, abs(point) * 0.35)
            lo, hi = point - half, point + half
        elif self.mode == "ratio":
            half = max(0.20, abs(point) * 0.08)
            lo, hi = point - half, point + half
        elif self.mode == "change_pct_oi":
            half = max(3.0, abs(point) * 0.40)
            lo, hi = point - half, point + half
        elif self.mode == "growth_pct":
            half = 5.0 if abs(point) < 1e-6 else max(5.0, abs(point) * 0.25)
            lo, hi = point - half, point + half
        elif self.mode == "return_pct":
            half = 5.0
            lo, hi = point - half, point + half
        elif self.mode == "percent":
            half = 0.5 if abs(point) < 1e-6 else max(0.5, abs(point) * 0.25)
            lo, hi = point - half, point + half
        elif self.kind == "classification":
            if "beat" in self.labels:
                band = max(abs(point) * 0.10, 0.15)
                lo, hi = point - band, point + band
            elif "direction" in self.name or self.labels == ("up", "down"):
                half = max(1.0, abs(point) * 0.02)
                lo, hi = point - half, point + half
            else:
                half = max(1.0, abs(point) * 0.10)
                lo, hi = point - half, point + half
        else:
            half = max(1.0, abs(point) * 0.25)
            lo, hi = point - half, point + half
        if self.lower is not None:
            lo = max(self.lower, lo)
        if self.upper is not None:
            hi = min(self.upper, hi)
        return lo, hi

    def fallback_point(self) -> float:
        point = 0.5 if self.mode == "probability" else 0.0
        if self.lower is not None:
            point = max(point, self.lower)
        if self.upper is not None:
            point = min(point, self.upper)
        return point

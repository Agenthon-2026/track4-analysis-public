"""Build an external, vintage-bound ALFRED development benchmark.

Each economic event has three target views, not three independent observations.
Downloaded snapshots and realized values never belong in the submission image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import subprocess
import time
import tomllib
import urllib.parse
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .dataset import REPO, load_cases, public_roots, require_external

SERIES = {f"DGS{years}": years for years in (1, 2, 3, 5, 7, 10, 20, 30)}
CPI_SERIES = {
    "CPIAUCSL": "All items CPI",
    "CPILFESL": "All items less food and energy CPI",
    "CPIUFDSL": "Food CPI",
    "CPIENGSL": "Energy CPI",
    "CUSR0000SAH1": "Shelter CPI",
    "CUSR0000SAS": "Services CPI",
}
FX_SERIES = {
    "DEXUSEU": ("EUR", "U.S. dollars per euro"),
    "DEXJPUS": ("JPY", "Japanese yen per U.S. dollar"),
    "DEXUSUK": ("GBP", "U.S. dollars per U.K. pound"),
}
ENERGY_SERIES = {
    "DCOILWTICO": ("WTI", "U.S. dollars per barrel, WTI at Cushing"),
    "DCOILBRENTEU": ("BRENT", "U.S. dollars per barrel, Brent Europe"),
    "DHHNGSP": ("HENRYHUB", "U.S. dollars per million BTU, Henry Hub natural gas"),
}
FAMILY_SERIES = {
    "curve_change": SERIES,
    "cpi_mom": CPI_SERIES,
    "fx_return": FX_SERIES,
    "energy_return": ENERGY_SERIES,
}
FAMILY_DOMAINS = {
    "curve_change": "rates",
    "cpi_mom": "inflation",
    "fx_return": "foreign-exchange",
    "energy_return": "energy",
}
KINDS = ("classification", "regression", "ranking")
SOURCE = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"


def _download(url: str) -> bytes:
    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            "30",
            "--retry",
            "1",
            "--retry-delay",
            "1",
            "--max-filesize",
            "8000000",
            url,
        ],
        check=False,
        capture_output=True,
        timeout=70,
    )
    if result.returncode:
        raise RuntimeError(
            f"ALFRED download failed ({result.returncode}): {result.stderr.decode(errors='replace')[:500]}"
        )
    return result.stdout


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def parse_snapshot(payload: bytes, series: str, vintage: date) -> dict[date, Decimal]:
    """The returned column must attest the requested vintage, not today's data."""
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    column = f"{series}_{vintage:%Y%m%d}"
    if reader.fieldnames != ["observation_date", column]:
        raise ValueError("ALFRED returned the wrong series/vintage or multiple columns")
    rows: dict[date, Decimal] = {}
    seen: set[date] = set()
    for row in reader:
        if set(row) != {"observation_date", column} or row[column] is None:
            raise ValueError("malformed ALFRED CSV row")
        day = date.fromisoformat(row["observation_date"])
        if day in seen or day > vintage:
            raise ValueError("duplicate or post-vintage ALFRED observation")
        seen.add(day)
        if row[column] in ("", "."):
            continue
        value = Decimal(row[column])
        if not value.is_finite() or not math.isfinite(float(value)):
            raise ValueError("nonfinite ALFRED observation")
        rows[day] = value
    if not rows:
        raise ValueError("ALFRED snapshot has no observations")
    return rows


def snapshot(
    cache: Path, series: str, vintage: date, *, offline: bool
) -> tuple[dict, dict]:
    if not any(series in roster for roster in FAMILY_SERIES.values()):
        raise ValueError("unsupported historical series")
    params = {
        "id": series,
        "cosd": (
            vintage - timedelta(days=400 if series in CPI_SERIES else 65)
        ).isoformat(),
        "coed": vintage.isoformat(),
        "vintage_date": vintage.isoformat(),
    }
    url = SOURCE + "?" + urllib.parse.urlencode(params)
    path = cache / f"{series}-{vintage}.csv"
    meta_path = path.with_suffix(".json")
    if path.exists() or meta_path.exists():
        payload = path.read_bytes()
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("url") != url
            or meta.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError("cached ALFRED snapshot identity/checksum mismatch")
    else:
        if offline:
            raise ValueError(f"missing frozen snapshot: {path.name}")
        payload = _download(url)
        if len(payload) > 8_000_000:
            raise ValueError("ALFRED response exceeds download budget")
        parse_snapshot(payload, series, vintage)
        meta = {
            "url": url,
            "series": series,
            "vintage": vintage.isoformat(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "license": "public-domain observations; citation requested",
            "series_metadata": f"https://fred.stlouisfed.org/series/{series}",
            "license_url": "https://fred.stlouisfed.org/legal/#copyright-public-domain",
            "publisher": "U.S. Energy Information Administration"
            if series in ENERGY_SERIES
            else "U.S. Bureau of Labor Statistics"
            if series in CPI_SERIES
            else "Board of Governors of the Federal Reserve System",
        }
        if series in ENERGY_SERIES or series in FX_SERIES:
            meta["publisher_reuse_terms"] = (
                "https://www.eia.gov/about/copyrights_reuse.php"
                if series in ENERGY_SERIES
                else "https://www.federalreserve.gov/disclaimer.htm"
            )
        path.write_bytes(payload)
        write_json(meta_path, meta)
        time.sleep(0.2)
    return parse_snapshot(payload, series, vintage), meta


def _toml(data: dict, prefix: str = "") -> str:
    """Serialize this card's scalar/list tables without editing TOML text."""
    lines = [f"[{prefix}]"] if prefix else []
    for key, value in data.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {json.dumps(value, allow_nan=False)}")
    for key, value in data.items():
        if isinstance(value, dict):
            lines += ["", _toml(value, f"{prefix}.{key}" if prefix else key)]
    return "\n".join(lines) + "\n"


def build_event(out: Path, event: dict, snapshots: dict) -> list[dict]:
    cutoff = date.fromisoformat(event["cutoff"])
    resolution = date.fromisoformat(event["resolution"])
    if resolution <= cutoff or (resolution - cutoff).days > 60:
        raise ValueError("event needs a positive horizon of at most 60 days")
    common_start = set.intersection(*(set(snapshots[(s, cutoff)][0]) for s in SERIES))
    common_end = set.intersection(*(set(snapshots[(s, resolution)][0]) for s in SERIES))
    start, end = max(common_start), max(common_end)
    if start > cutoff or end > resolution or end <= cutoff:
        raise ValueError("event snapshots do not bound a future observation")
    if (cutoff - start).days > 7 or (resolution - end).days > 7:
        raise ValueError("stale snapshot: no common recent curve")
    entities, outcomes, documents, provenance = [], [], {}, []
    for series, years in SERIES.items():
        before, before_meta = snapshots[(series, cutoff)]
        after, after_meta = snapshots[(series, resolution)]
        entity_id = f"UST{years}Y"
        history = sorted(d for d in before if cutoff - timedelta(days=35) <= d <= start)
        # Keep the most recent observations, with a short earlier context, in a bounded table.
        selected = sorted(set(history[::5] + history[-5:]))
        doc_id = f"ALFRED_{series}_{cutoff:%Y%m%d}"
        title = f"{years}-Year U.S. Treasury constant-maturity yield"
        text = (
            f"{entity_id}: {title}. {series} observations in percent, available in the "
            f"ALFRED vintage dated {cutoff}. This is a historical table, not a future forecast.\n"
            f"date | {series}\n"
            + "\n".join(f"{day} | {before[day]}" for day in selected)
        )
        entities.append(
            {
                "entity_id": entity_id,
                "name": title,
                "series_fred": series,
                "maturity_years": years,
                "start_yield_pct": float(before[start]),
                "as_of": start.isoformat(),
                "unit": "bps_change",
            }
        )
        change = float((after[end] - before[start]) * 100)
        outcomes.append(
            {
                "entity_id": entity_id,
                "y": change,
                "true_label": "up" if change > 5 else "down" if change < -5 else "flat",
            }
        )
        documents[doc_id] = {
            "doc_id": doc_id,
            "doc_date": cutoff.isoformat(),
            "source": before_meta["url"],
            "text": text,
        }
        provenance.append(
            {"series": series, "input": before_meta, "outcome": after_meta}
        )
    prompt = (
        "Using only the frozen table and corpus, predict the cross-section of Treasury "
        "yield changes in basis points. The target is 100 * (the latest common yield "
        f"available in the ALFRED vintage of {resolution} minus start_yield_pct). "
        "Each starting yield is supplied. Do not treat historical observations as known future values. "
        "Provide a numeric point_forecast, a 90% interval on the basis-point change scale, and citations. "
    )
    return _write_views(
        out,
        event,
        entities,
        outcomes,
        documents,
        provenance,
        start=start,
        end=end,
        slug="curve",
        family="local_curve_change",
        title="Independent local Treasury cross-section",
        tag="rates",
        target_name="yield_change_bps",
        unit_name="bps_change",
        prompt=prompt,
        classification="Labels: up if change > 5 bps, down if change < -5 bps, flat otherwise.",
        ranking="Rank larger yield changes higher; point_forecast is the predicted change, not a rank integer.",
        limitations="One rates domain with three correlated target views; not representative of hidden T4 families. Model training contamination is not certified.",
    )


def build_market_event(
    out: Path, event: dict, snapshots: dict, *, family: str
) -> list[dict]:
    """Quote-level changes measured from the known cutoff reference, never a revised one."""
    if family not in ("fx_return", "energy_return"):
        raise ValueError("unsupported market-return family")
    roster = FAMILY_SERIES[family]
    cutoff = date.fromisoformat(event["cutoff"])
    resolution = date.fromisoformat(event["resolution"])
    if not 0 < (resolution - cutoff).days <= 60:
        raise ValueError("market event needs a positive horizon of at most 60 days")
    starts = set.intersection(
        *(set(snapshots[(series, cutoff)][0]) for series in roster)
    )
    ends = set.intersection(
        *(set(snapshots[(series, resolution)][0]) for series in roster)
    )
    if not starts or not ends:
        raise ValueError("market snapshots have no common observations")
    start, end = max(starts), max(ends)
    if not start <= cutoff < end <= resolution:
        raise ValueError(
            "market snapshots must resolve an observation after the cutoff"
        )
    if (cutoff - start).days > 7 or (resolution - end).days > 7:
        raise ValueError("stale market snapshot")
    entities, outcomes, documents, sources = [], [], {}, []
    for series, (entity_id, quote_unit) in roster.items():
        before, before_meta = snapshots[(series, cutoff)]
        after, after_meta = snapshots[(series, resolution)]
        if before[start] <= 0:
            raise ValueError("market return requires a positive cutoff reference")
        history = sorted(
            day for day in before if cutoff - timedelta(days=35) <= day <= start
        )
        if len(history) < 5:
            raise ValueError("market snapshot has insufficient recent context")
        selected = sorted(set(history[::5] + history[-5:]))
        doc_id = f"ALFRED_{series}_{cutoff:%Y%m%d}"
        text = (
            f"{entity_id}: {series}, quoted in {quote_unit}. Values available in the ALFRED "
            f"vintage dated {cutoff}. These are historical observations, not future forecasts.\n"
            f"date | {series}\n"
            + "\n".join(f"{day} | {before[day]}" for day in selected)
        )
        entities.append(
            {
                "entity_id": entity_id,
                "name": quote_unit,
                "series_fred": series,
                "start_level": float(before[start]),
                "as_of": start.isoformat(),
                "quote_unit": quote_unit,
                "unit": "percent",
            }
        )
        change = float((after[end] / before[start] - 1) * 100)
        outcomes.append(
            {
                "entity_id": entity_id,
                "y": change,
                "true_label": "up" if change > 0 else "down" if change < 0 else "flat",
            }
        )
        documents[doc_id] = {
            "doc_id": doc_id,
            "doc_date": cutoff.isoformat(),
            "source": before_meta["url"],
            "text": text,
        }
        sources.append({"series": series, "input": before_meta, "outcome": after_meta})
    domain = FAMILY_DOMAINS[family]
    prompt = (
        f"Using only the frozen table and corpus, predict {domain} quote-level changes. "
        "The target is 100 * (the latest common quote level available in the ALFRED vintage "
        f"of {resolution} / start_level - 1). The starting reference is frozen at cutoff; "
        "do not substitute a revised starting value from a later vintage. Quotation units and "
        "directions are stated per entity; do not invert exchange-rate quotes. "
        "The outcome observation must be after cutoff. Publication can lag the observation date. "
        "Provide a numeric point_forecast and a 90% interval in percent, with citations. "
        "Historical quotes are inputs, not known future outcomes. "
    )
    return _write_views(
        out,
        event,
        entities,
        outcomes,
        documents,
        sources,
        start=start,
        end=end,
        slug=family.removesuffix("_return"),
        family=f"local_{family}_vintage",
        title=f"Local {domain} quote-level cross-section",
        tag=domain,
        target_name=f"{family}_pct",
        unit_name="percent",
        prompt=prompt,
        classification="Labels: up if change > 0 percent, down if change < 0 percent, flat otherwise.",
        ranking="Rank larger quote-level percentage changes higher; point_forecast is the change, not a rank integer.",
        limitations="Specified-vintage quotes with publication delays, not certified first-release values. Entity quotes and target views are correlated; each window is one event group. Separate domains may share market shocks. Input history is not a future forecast or production NLI evidence.",
    )


def _previous_month(month: date) -> date:
    return (month - timedelta(days=1)).replace(day=1)


def build_cpi_event(out: Path, event: dict, snapshots: dict) -> list[dict]:
    cutoff = date.fromisoformat(event["cutoff"])
    resolution = date.fromisoformat(event["resolution"])
    month = date.fromisoformat(event["target_month"])
    prior = _previous_month(month)
    entities, outcomes, documents, provenance = [], [], {}, []
    for series, title in CPI_SERIES.items():
        before, before_meta = snapshots[(series, cutoff)]
        after, after_meta = snapshots[(series, resolution)]
        if any(
            day.day != 1 or value <= 0
            for rows in (before, after)
            for day, value in rows.items()
        ):
            raise ValueError("CPI snapshots require positive monthly index levels")
        if month in before or max(before) != prior:
            raise ValueError(
                "CPI target must be the next unpublished observation month"
            )
        if max(after) != month or prior not in after:
            raise ValueError(
                "CPI resolution vintage must contain the target and previous month"
            )
        previous = _previous_month(prior)
        if previous not in before:
            raise ValueError("CPI input needs consecutive months for historical change")
        historical_change = float((before[prior] / before[previous] - 1) * 100)
        change = float((after[month] / after[prior] - 1) * 100)
        doc_id = f"ALFRED_{series}_{cutoff:%Y%m%d}"
        text = (
            f"{series}: {title}, seasonally adjusted index levels from BLS CPI. "
            f"ALFRED vintage {cutoff}; observation dates identify months, not release dates. "
            "This is historical evidence, not a forecast.\n"
            f"{series} historical month-over-month change for {prior}: {historical_change:.12g} percent.\n"
            f"date | {series}\n"
        ) + "\n".join(f"{day} | {before[day]}" for day in sorted(before)[-7:])
        entities.append(
            {
                "entity_id": series,
                "name": title,
                "series_fred": series,
                "last_mom_pct": historical_change,
                "as_of": cutoff.isoformat(),
                "last_observation_month": prior.isoformat(),
                "unit": "percent",
            }
        )
        outcomes.append(
            {
                "entity_id": series,
                "y": change,
                "true_label": "up" if change > 0 else "down" if change < 0 else "flat",
            }
        )
        documents[doc_id] = {
            "doc_id": doc_id,
            "doc_date": cutoff.isoformat(),
            "source": before_meta["url"],
            "text": text,
        }
        provenance.append(
            {"series": series, "input": before_meta, "outcome": after_meta}
        )
    prompt = (
        "Using only the frozen table and corpus, forecast seasonally adjusted CPI "
        f"month-over-month changes for observation month {month}. "
        f"The target is 100 * (index[{month}] / index[{prior}] - 1), with both index "
        f"levels taken from the ALFRED vintage dated {resolution}. "
        "This is a specified-vintage target, not a certified first-release target. "
        "Provide a numeric point_forecast and a 90% interval in percent, with citations. "
        "Historical changes are inputs, not known future outcomes. "
    )
    return _write_views(
        out,
        event,
        entities,
        outcomes,
        documents,
        provenance,
        start=prior,
        end=month,
        slug="cpi",
        family="local_cpi_mom_vintage",
        title="Independent local CPI component cross-section",
        tag="inflation",
        target_name="cpi_component_mom_vintage_pct",
        unit_name="percent",
        prompt=prompt,
        classification="Labels: up if change > 0, down if change < 0, flat otherwise.",
        ranking="Rank larger month-over-month changes higher; point_forecast is the change, not a rank integer.",
        limitations="Specified-vintage CPI, not certified first print. Components overlap and views share one release event; neither rows nor views are independent samples. Model training contamination is not certified.",
    )


def _write_views(
    out: Path,
    event: dict,
    entities: list,
    outcomes: list,
    documents: dict,
    provenance: list,
    *,
    start: date,
    end: date,
    slug: str,
    family: str,
    title: str,
    tag: str,
    target_name: str,
    unit_name: str,
    prompt: str,
    classification: str,
    ranking: str,
    limitations: str,
) -> list[dict]:
    cutoff = date.fromisoformat(event["cutoff"])
    resolution = date.fromisoformat(event["resolution"])
    cases = []
    for kind in KINDS:
        case_id = f"t4-local-{slug}-{cutoff:%Y%m%d}-{kind}"
        unit = out / "inputs" / case_id
        (unit / "corpus").mkdir(parents=True)
        view_prompt = prompt + (
            {"classification": classification, "ranking": ranking}.get(kind, "")
        )
        target = {"name": target_name, "type": kind, "unit": unit_name}
        if kind == "classification":
            target["labels"] = ["down", "flat", "up"]
        task = {
            "schema_version": "3",
            "task_id": case_id,
            "family": family,
            "cutoff_date": cutoff.isoformat(),
            "resolution_date": resolution.isoformat(),
            "target": target,
            "interval_level": 0.9,
            "prompt": view_prompt,
            "entities": entities,
        }
        write_json(unit / "task.json", task)
        for doc_id, doc in documents.items():
            write_json(unit / "corpus" / f"{doc_id}.json", doc)
        card = tomllib.loads((REPO / "units/t4-EXAMPLE-eps-beat/card.toml").read_text())
        card["task"].update(
            id=case_id,
            title=title,
            split="public-dev",
            family=family,
            prompt=view_prompt,
            cutoff_date=cutoff.isoformat(),
            resolution_date=resolution.isoformat(),
            target_type=kind,
            adversarial=False,
            faithfulness_rubric="Citations must support the submitted prediction using only the declared vintage corpus; local smoke does not test production NLI.",
        )
        card["metadata"].update(
            author_name="Local development",
            author_email="local@example.invalid",
            category=family,
            tags=["analysis", "local-development", tag],
        )
        card["provenance"].update(
            data_source="ALFRED vintage snapshots",
            data_cutoff=cutoff.isoformat(),
            license="public-domain",
        )
        card["contamination"]["canary_guid"] = str(
            uuid.UUID(
                bytes=hashlib.sha256(("local-evaluation/" + case_id).encode()).digest()[
                    :16
                ],
                version=4,
            )
        )
        card["scoring"]["params"]["target_type"] = kind
        (unit / "card.toml").write_text(_toml(card))
        files = []
        for path in sorted([unit / "task.json", *unit.glob("corpus/*.json")]):
            payload = path.read_bytes()
            files.append(
                {
                    "path": path.relative_to(unit).as_posix(),
                    "role": "task" if path.name == "task.json" else "corpus",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "source": "ALFRED vintage snapshots",
                    "license": "public-domain",
                    "redistributable": True,
                    "pii_stripped": True,
                    "cutoff": cutoff.isoformat(),
                }
            )
        write_json(
            unit / "manifest.json",
            {"manifest_version": "2.0", "unit_id": case_id, "files": files},
        )
        truth = out / "truth" / f"{case_id}.json"
        write_json(
            truth,
            {
                "unit_id": case_id,
                "cutoff_date": cutoff.isoformat(),
                "target_type": kind,
                "outcomes": outcomes,
            },
        )
        cases.append(
            {
                "id": case_id,
                "unit_dir": f"inputs/{case_id}",
                "truth_path": f"truth/{case_id}.json",
                "split": event["split"],
                "group": event["id"],
                "domain": tag,
            }
        )
    write_json(
        out / "provenance" / f"{event['id']}.json",
        {
            "event": event,
            "start_observation": start.isoformat(),
            "end_observation": end.isoformat(),
            "sources": provenance,
            "independent_events": 1,
            "limitations": limitations,
        },
    )
    return cases


def build(spec_path: Path, cache: Path, out: Path, *, offline: bool = False) -> Path:
    roots = public_roots()
    for path in (spec_path, cache, out):
        require_external(path, roots)
    spec = json.loads(spec_path.read_text())
    family = spec.get("family", "curve_change")
    if family not in FAMILY_SERIES:
        raise ValueError("unsupported historical family")
    series_roster = FAMILY_SERIES[family]
    events = spec.get("events")
    if spec.get("version") != 1 or not isinstance(events, list) or not events:
        raise ValueError("expected version 1 and a nonempty events roster")
    seen = set()
    periods: dict[str, list[tuple[date, date]]] = {}
    for event in events:
        fields = {"id", "cutoff", "resolution", "split"}
        if family == "cpi_mom":
            fields.add("target_month")
        if set(event) != fields:
            raise ValueError("invalid event fields")
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", event["id"])
            or event["id"] in seen
        ):
            raise ValueError("invalid or duplicate event id")
        seen.add(event["id"])
        if event["split"] not in ("train", "calibration", "test"):
            raise ValueError("invalid event split")
        cutoff, resolution = (
            date.fromisoformat(event["cutoff"]),
            date.fromisoformat(event["resolution"]),
        )
        if not 0 < (resolution - cutoff).days <= 60:
            raise ValueError("event needs a positive horizon of at most 60 days")
        if resolution >= date.today():
            raise ValueError("historical event must already be resolved")
        if family == "cpi_mom":
            month = date.fromisoformat(event["target_month"])
            if month.day != 1 or month > cutoff or (cutoff - month).days > 62:
                raise ValueError(
                    "CPI target_month must identify a recent observation month"
                )
        periods.setdefault(event["split"], []).append((cutoff, resolution))
    splits = ("train", "calibration", "test")
    for i, earlier in enumerate(splits):
        for later in splits[i + 1 :]:
            if (
                earlier in periods
                and later in periods
                and max(end for _, end in periods[earlier])
                >= min(start for start, _ in periods[later])
            ):
                raise ValueError("earlier outcomes must precede later cutoffs")
    ordered = sorted(
        (date.fromisoformat(e["cutoff"]), date.fromisoformat(e["resolution"]))
        for e in events
    )
    if any(left[1] >= right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("event windows must not overlap")
    if out.exists():
        raise ValueError("benchmark output must be new")
    cache.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for event in events:
        for key in ("cutoff", "resolution"):
            vintage = date.fromisoformat(event[key])
            missing = [
                series for series in series_roster if (series, vintage) not in snapshots
            ]
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = {
                    series: pool.submit(
                        snapshot, cache, series, vintage, offline=offline
                    )
                    for series in missing
                }
                for series, job in jobs.items():
                    try:
                        snapshots[(series, vintage)] = job.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Snapshot {series} at {vintage}: {exc}"
                        ) from exc
            print(f"Validated vintage {vintage}", flush=True)
    out.mkdir(parents=True)
    for folder in ("truth", "provenance"):
        (out / folder).mkdir()
    if family in ("fx_return", "energy_return"):
        cases = [
            case
            for event in events
            for case in build_market_event(out, event, snapshots, family=family)
        ]
    else:
        builder = build_cpi_event if family == "cpi_mom" else build_event
        cases = [case for event in events for case in builder(out, event, snapshots)]
    manifest = out / "manifest.json"
    write_json(manifest, {"version": 1, "cases": cases})
    load_cases(units=None, manifest=manifest)
    from qfbench2_common import manifest as manifests, taskcard

    for case in cases:
        unit = out / case["unit_dir"]
        _, errors = taskcard.load_and_validate(unit)
        errors += manifests.verify_manifest(unit)
        errors += manifests.assert_public_safe(unit)
        if errors:
            raise ValueError(
                f"generated unit failed validation: {case['id']}: {errors}"
            )
    write_json(
        out / "build.json",
        {
            "version": 1,
            "complete": True,
            "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "events": len(events),
            "cases": len(cases),
            "entities_per_case": len(series_roster),
            "official_score": None,
            "domain": FAMILY_DOMAINS[family],
            "family": family,
        },
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(build(args.spec, args.cache, args.out, offline=args.offline))
        return 0
    except Exception as exc:
        print(f"Historical build aborted: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

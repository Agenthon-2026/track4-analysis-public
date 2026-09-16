"""Historical builders reject wrong vintages and keep truth out of inputs."""

import json
from datetime import date, timedelta
from decimal import Decimal

import pytest

from baselines.evaluation import historical


def test_exact_single_series_vintage_required():
    data = b"observation_date,DGS2_20250115\n2025-01-13,4.25\n2025-01-14,\n"
    assert historical.parse_snapshot(data, "DGS2", date(2025, 1, 15)) == {
        date(2025, 1, 13): Decimal("4.25")
    }
    with pytest.raises(ValueError, match="wrong series/vintage"):
        historical.parse_snapshot(
            data.replace(b"20250115", b"20260908"), "DGS2", date(2025, 1, 15)
        )
    with pytest.raises(ValueError, match="wrong series/vintage"):
        historical.parse_snapshot(
            b"observation_date,DGS1_20250115,DGS2_20260908\n", "DGS1", date(2025, 1, 15)
        )


@pytest.mark.parametrize(
    "rows",
    [
        "2025-01-16,4.0\n",
        "2025-01-13,4\n2025-01-13,4\n",
        "2025-01-13,NaN\n",
        "2025-01-13,Infinity\n",
        "2025-01-13,4,unexpected\n",
    ],
)
def test_invalid_observations_refused(rows):
    with pytest.raises(ValueError):
        historical.parse_snapshot(
            ("observation_date,DGS2_20250115\n" + rows).encode(),
            "DGS2",
            date(2025, 1, 15),
        )


def test_tampered_cache_is_not_refetched(tmp_path, monkeypatch):
    path = tmp_path / "DGS2-2025-01-15.csv"
    path.write_text("changed")
    path.with_suffix(".json").write_text(
        json.dumps({"url": "wrong", "sha256": "wrong"})
    )
    monkeypatch.setattr(
        historical,
        "_download",
        lambda *a, **k: pytest.fail("must not redownload tampered cache"),
    )
    with pytest.raises(ValueError, match="checksum"):
        historical.snapshot(tmp_path, "DGS2", date(2025, 1, 15), offline=False)


def test_overlap_refused_before_fetch(tmp_path, monkeypatch):
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "version": 1,
                "events": [
                    {
                        "id": "a",
                        "cutoff": "2025-01-01",
                        "resolution": "2025-02-01",
                        "split": "train",
                    },
                    {
                        "id": "b",
                        "cutoff": "2025-02-01",
                        "resolution": "2025-03-01",
                        "split": "test",
                    },
                ],
            }
        )
    )
    monkeypatch.setattr(
        historical, "snapshot", lambda *a, **k: pytest.fail("must reject before fetch")
    )
    with pytest.raises(ValueError, match="precede"):
        historical.build(spec, tmp_path / "cache", tmp_path / "out")


def test_builder_produces_three_correlated_views_without_future_inputs(
    tmp_path, monkeypatch
):
    pytest.importorskip("qfbench2_common")
    from baselines.evaluation.dataset import load_cases, stage_inputs
    from baselines.evaluation.assess import assess_unit

    cutoff, resolution = date(2025, 1, 15), date(2025, 2, 4)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "version": 1,
                "events": [
                    {
                        "id": "invented-event",
                        "cutoff": str(cutoff),
                        "resolution": str(resolution),
                        "split": "test",
                    }
                ],
            }
        )
    )

    def invented(cache, series, vintage, **kwargs):
        base = Decimal(historical.SERIES[series]) / 10
        rows = {
            vintage - timedelta(days=i): base + Decimal("0.01") * i
            for i in range(1, 30)
        }
        return rows, {
            "url": "https://example.invalid/synthetic",
            "sha256": "synthetic",
            "vintage": str(vintage),
        }

    monkeypatch.setattr(historical, "snapshot", invented)
    out = tmp_path / "built"
    roster = historical.build(spec, tmp_path / "cache", out)
    cases = load_cases(units=None, manifest=roster)
    assert len(cases) == 3 and len({c.group for c in cases}) == 1
    for case in cases:
        staged = tmp_path / case.case_id
        stage_inputs(case.unit_dir, staged)
        assert set(p.name for p in staged.iterdir()) == {"task.json", "corpus"}
        assert not case.truth_path.is_relative_to(case.unit_dir)
        text = " ".join(p.read_text() for p in (staged / "corpus").glob("*.json"))
        assert str(resolution) not in text
        truth = json.loads(case.truth_path.read_text())
        assert len(truth["outcomes"]) == 8
        output = tmp_path / (case.case_id + "-output")
        output.mkdir()
        # Invalid participant output still retains the official worst-case development score.
        result = assess_unit(case.unit_dir, output, realized=truth)
        assert result["development_score"] == pytest.approx(-0.27)
    with pytest.raises(ValueError, match="must be new"):
        historical.build(spec, tmp_path / "cache", out, offline=True)


def cpi_event():
    return {
        "id": "invented-cpi",
        "cutoff": "2025-02-05",
        "resolution": "2025-02-20",
        "target_month": "2025-01-01",
        "split": "calibration",
    }


def cpi_snapshot(cache, series, vintage, **kwargs):
    if str(vintage) == cpi_event()["cutoff"]:
        rows = {date(2024, 11, 1): Decimal("100"), date(2024, 12, 1): Decimal("110")}
    else:
        # The denominator was revised: use both levels from the resolution vintage.
        rows = {date(2024, 12, 1): Decimal("200"), date(2025, 1, 1): Decimal("206")}
    return rows, {"url": "https://example.invalid/synthetic", "vintage": str(vintage)}


def test_cpi_builder_uses_resolution_denominator_and_only_historical_inputs(
    tmp_path, monkeypatch
):
    pytest.importorskip("qfbench2_common")
    from baselines.evaluation.dataset import load_cases, stage_inputs
    from baselines.strong_rag_baseline.indexer import build_index
    from baselines.strong_rag_baseline.reasoner import ground_entity
    from baselines.strong_rag_baseline.retriever import BM25Index

    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"version": 1, "family": "cpi_mom", "events": [cpi_event()]})
    )
    monkeypatch.setattr(historical, "snapshot", cpi_snapshot)
    roster = historical.build(spec, tmp_path / "cache", tmp_path / "built")
    cases = load_cases(units=None, manifest=roster)
    assert len(cases) == 3 and len({case.group for case in cases}) == 1
    for case in cases:
        task = json.loads((case.unit_dir / "task.json").read_text())
        truth = json.loads(case.truth_path.read_text())
        assert [row["y"] for row in truth["outcomes"]] == [3.0] * len(
            historical.CPI_SERIES
        )
        assert task["family"] == "local_cpi_mom_vintage"
        assert all(row["last_mom_pct"] == 10 for row in task["entities"])
        assert "not a certified first-release" in task["prompt"]
        staged = tmp_path / case.case_id
        stage_inputs(case.unit_dir, staged)
        assert set(p.name for p in staged.iterdir()) == {"task.json", "corpus"}
        text = " ".join(p.read_text() for p in (staged / "corpus").glob("*.json"))
        assert "206" not in text and "2025-02-20" not in text
        corpus = build_index(staged / "corpus")
        index = BM25Index(corpus.chunks, task["cutoff_date"])
        for entity in task["entities"]:
            result = ground_entity(task, entity, corpus, index)
            assert result.point == 10 and result.claims


@pytest.mark.parametrize(
    "defect",
    [
        "already_published",
        "stale",
        "missing_previous",
        "wrong_month",
        "non_monthly",
        "zero_index",
    ],
)
def test_cpi_rejects_unsettled_or_leaking_snapshots(tmp_path, defect):
    event = cpi_event()
    snapshots = {
        (series, date.fromisoformat(event[key])): cpi_snapshot(
            None, series, date.fromisoformat(event[key])
        )
        for series in historical.CPI_SERIES
        for key in ("cutoff", "resolution")
    }
    before = snapshots[("CPIAUCSL", date.fromisoformat(event["cutoff"]))][0]
    after = snapshots[("CPIAUCSL", date.fromisoformat(event["resolution"]))][0]
    if defect == "already_published":
        before[date(2025, 1, 1)] = Decimal("115")
    elif defect == "stale":
        del before[date(2024, 12, 1)]
    elif defect == "missing_previous":
        del after[date(2024, 12, 1)]
    elif defect == "wrong_month":
        after[date(2025, 2, 1)] = Decimal("207")
    elif defect == "non_monthly":
        before[date(2024, 11, 2)] = Decimal("101")
    else:
        after[date(2024, 12, 1)] = Decimal("0")
    with pytest.raises(ValueError, match="CPI"):
        historical.build_cpi_event(tmp_path, event, snapshots)


def test_cpi_invalid_month_rejected_before_download(tmp_path, monkeypatch):
    event = cpi_event() | {"target_month": "2025-01-02"}
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"version": 1, "family": "cpi_mom", "events": [event]}))
    monkeypatch.setattr(
        historical, "snapshot", lambda *a, **k: pytest.fail("must reject before fetch")
    )
    with pytest.raises(ValueError, match="target_month"):
        historical.build(spec, tmp_path / "cache", tmp_path / "built")


def market_event():
    return {
        "id": "invented-market-window",
        "cutoff": "2025-01-15",
        "resolution": "2025-01-29",
        "split": "train",
    }


def market_snapshots(family):
    cutoff = date.fromisoformat(market_event()["cutoff"])
    resolution = date.fromisoformat(market_event()["resolution"])
    result = {}
    for i, series in enumerate(historical.FAMILY_SERIES[family]):
        base = Decimal(100 * (i + 1))
        before = {cutoff - timedelta(days=j): base - j + 1 for j in range(1, 30)}
        # The later vintage revises the old level. It must not replace the known reference.
        after = {
            cutoff - timedelta(days=1): Decimal("9999"),
            resolution - timedelta(days=1): base
            * (Decimal("1.1") - Decimal("0.1") * i),
        }
        for vintage, rows in ((cutoff, before), (resolution, after)):
            result[(series, vintage)] = (
                rows,
                {"url": "https://example.invalid/synthetic", "vintage": str(vintage)},
            )
    return result


@pytest.mark.parametrize("family", ["fx_return", "energy_return"])
def test_market_views_bind_known_reference_domain_and_past_only_inputs(
    tmp_path, monkeypatch, family
):
    pytest.importorskip("qfbench2_common")
    from baselines.evaluation.dataset import load_cases, stage_inputs

    snapshots = market_snapshots(family)
    monkeypatch.setattr(
        historical,
        "snapshot",
        lambda cache, series, vintage, **kwargs: snapshots[(series, vintage)],
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"version": 1, "family": family, "events": [market_event()]})
    )
    roster = historical.build(spec, tmp_path / "cache", tmp_path / "built")
    cases = load_cases(units=None, manifest=roster)
    assert {case.domain for case in cases} == {historical.FAMILY_DOMAINS[family]}
    assert {case.group for case in cases} == {market_event()["id"]}
    assert len(cases) == 3
    for case in cases:
        task = json.loads((case.unit_dir / "task.json").read_text())
        truth = json.loads(case.truth_path.read_text())
        assert [row["y"] for row in truth["outcomes"]] == [10, 0, -10]
        assert [row["start_level"] for row in task["entities"]] == [100, 200, 300]
        assert [row["true_label"] for row in truth["outcomes"]] == [
            "up",
            "flat",
            "down",
        ]
        assert ("labels" in task["target"]) == (
            task["target"]["type"] == "classification"
        )
        staged = tmp_path / case.case_id
        stage_inputs(case.unit_dir, staged)
        text = " ".join(p.read_text() for p in (staged / "corpus").glob("*.json"))
        assert "9999" not in text and "2025-01-28" not in text
        assert not case.truth_path.is_relative_to(staged)
        assert {p.name for p in staged.iterdir()} == {"task.json", "corpus"}
        if family == "fx_return":
            assert task["entities"][1]["quote_unit"] == "Japanese yen per U.S. dollar"


@pytest.mark.parametrize(
    "defect",
    [
        "past_outcome",
        "stale_input",
        "missing_common",
        "zero_reference",
        "short_history",
        "future_vintage",
    ],
)
def test_market_invalid_windows_are_not_silently_dropped(tmp_path, defect):
    snapshots = market_snapshots("fx_return")
    cutoff, resolution = date(2025, 1, 15), date(2025, 1, 29)
    before = snapshots[("DEXUSEU", cutoff)][0]
    after = snapshots[("DEXUSEU", resolution)][0]
    if defect == "past_outcome":
        after.pop(resolution - timedelta(days=1))
    elif defect == "stale_input":
        for day in list(before):
            if (cutoff - day).days <= 7:
                del before[day]
    elif defect == "missing_common":
        before.clear()
    elif defect == "zero_reference":
        before[cutoff - timedelta(days=1)] = Decimal(0)
    elif defect == "short_history":
        for day in list(before):
            if (cutoff - day).days > 2:
                del before[day]
    else:
        for series in historical.FX_SERIES:
            snapshots[(series, resolution)][0][resolution + timedelta(days=1)] = (
                Decimal(100)
            )
    with pytest.raises(ValueError, match="market"):
        historical.build_market_event(
            tmp_path, market_event(), snapshots, family="fx_return"
        )


def test_market_overlapping_events_rejected_before_fetch(tmp_path, monkeypatch):
    spec = tmp_path / "spec.json"
    second = market_event() | {
        "id": "overlapping",
        "cutoff": "2025-01-28",
        "resolution": "2025-02-10",
    }
    spec.write_text(
        json.dumps(
            {"version": 1, "family": "fx_return", "events": [market_event(), second]}
        )
    )
    monkeypatch.setattr(
        historical,
        "snapshot",
        lambda *a, **k: pytest.fail("must refuse before fetching"),
    )
    with pytest.raises(ValueError, match="windows must not overlap"):
        historical.build(spec, tmp_path / "cache", tmp_path / "built")

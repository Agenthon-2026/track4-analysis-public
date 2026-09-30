"""The number-form scan is a standing check: every corpus the repository ships reads cleanly.

`scripts/scan_number_forms.py` finds every numeric expression of the risky kinds (fractions,
numbers in words, index bases, dates, ranges, footnote markers, glued suffixes, and any other token
the figure reader reads nothing from) and compares the scorer's reading with what the expression
means. These tests run it over the shipped units, and over planted texts, so a unit that brings a
form the matcher cannot read fails here instead of silently penalising participants.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
from types import ModuleType

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "scan_number_forms", ROOT / "scripts" / "scan_number_forms.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scan = _load()


def _verdicts(text: str) -> dict[tuple[str, str], set[str]]:
    report = scan.Report()
    scan.scan_text(text, report)
    out: dict[tuple[str, str], set[str]] = {}
    for (family, kind, verdict, _), _count in report.counts.items():
        out.setdefault((family, kind), set()).add(verdict)
    return out


def _plant(tmp_path: pathlib.Path, text: str) -> pathlib.Path:
    unit = tmp_path / "t4-planted"
    (unit / "corpus").mkdir(parents=True)
    doc = {"doc_id": "DOC", "doc_date": "2031-07-19", "text": text}
    (unit / "corpus" / "DOC.json").write_text(json.dumps(doc), encoding="utf-8")
    return unit


def test_every_shipped_unit_reads_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    units = scan.unit_dirs_under([ROOT / "units"]) if (ROOT / "units").is_dir() else []
    if not units:
        pytest.skip("no unit corpora in this checkout; the scan reads them at run time")
    report = scan.scan_units(units)
    assert report.documents > len(units)
    gaps = {k: n for k, n in report.counts.items() if k[2] == "gap"}
    assert not gaps, f"number forms the matcher misreads: {gaps}"


@pytest.mark.parametrize(
    ("text", "family", "kind"),
    [
        ("Velmora trimmed 1/4 percentage point", "fraction", "simple"),
        ("Brimco needs 66⅔%", "fraction", "mixed"),
        ("Velmora sees two quarter-point trims", "fraction_word", "quarter"),
        ("Zent slid four basis points", "spelled", "point"),
        ("Brimco (1982-84=100)", "index_base", "year_base"),
        ("Zent on 3/20", "date", "m_d"),
        ("Brimco band 4.25-4.50", "range", "amounts"),
        ("Zent ratio 12.5%1 here", "footnote", "percent_digit"),
        ("leverage of 7.3x", "glued", "suffix_x"),
        ("returned $212mm", "glued", "currency_mm"),
        ("12,345.6Total Loans", "glued", "value_then_label"),
    ],
)
def test_each_family_finds_its_form_and_the_matcher_covers_it(
    text: str, family: str, kind: str
) -> None:
    assert _verdicts(text).get((family, kind)) == {"covered"}


@pytest.mark.parametrize(
    ("text", "family", "kind"),
    [
        pytest.param(
            "Velmora trimmed 3/16 percentage point",
            "fraction",
            "simple",
            id="sixteenths",
        ),
        pytest.param("a 5/12 point move", "fraction", "simple", id="twelfths"),
    ],
)
def test_an_unreadable_form_is_a_gap(text: str, family: str, kind: str) -> None:
    assert "gap" in _verdicts(text).get((family, kind), set())


def test_a_token_no_class_explains_is_a_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every unread token must be explained by a named class; one that is not fails the scan."""
    assert _verdicts("the 10-K filing")[("unread", "id_with_punctuation")] == {
        "left_alone"
    }
    monkeypatch.setattr(scan, "_UNREAD_CLASSES", ())
    assert _verdicts("the 10-K filing")[("unread", "unclassified")] == {"gap"}


def test_a_planted_unit_with_an_unreadable_form_fails_the_scan(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    unit = _plant(tmp_path, "Planted sentinel: Velmora trimmed 3/16 percentage point.")
    assert scan.main([str(unit)]) == 1
    out = capsys.readouterr().out
    assert "gap" in out and "fraction/simple" in out
    # Counts only by default: corpus text of a held-out unit is private.
    assert "sentinel" not in out
    assert scan.main([str(unit), "--show", "gap"]) == 1
    assert "sentinel" in capsys.readouterr().out


def test_a_clean_planted_unit_passes(tmp_path: pathlib.Path) -> None:
    unit = _plant(tmp_path, "Velmora trimmed 1/4 percentage point on 3/20.")
    assert scan.main([str(unit), "--json"]) == 0


def test_a_rule_the_scan_relies_on_is_what_the_scan_measures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scan reads with the scorer's own `figures`: with the fraction rule removed, the same
    text becomes a gap."""
    from qfbench2_track_analysis import numeric

    text = "Velmora trimmed 1/4 percentage point"
    assert _verdicts(text)[("fraction", "simple")] == {"covered"}
    monkeypatch.setattr(numeric, "_SIMPLE_FRACTION", re.compile(r"(?!x)x(\d)?(\d)?"))
    assert _verdicts(text)[("fraction", "simple")] == {"gap"}

"""Unit tests for the local ``submitted_reasons`` check (pure stdlib, no network).

The caps and the deny list are the published reasoning contract (SUBMISSION_CLI.md, "How
reasoning is scored"). Each cap is tested at its boundary and one past it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from baselines.guardrails_example.citation_rail import (
    CorpusDoc,
    check_submitted_reasons,
    load_corpus,
)

REPO = Path(__file__).resolve().parents[3]
CUTOFF = "2024-03-15"
LONG = CorpusDoc(doc_id="long", text="a" * 60_000, doc_date="2024-02-01")
WIDE = CorpusDoc(doc_id="wide", text="é" * 8_000, doc_date="2024-02-01")  # two bytes each
QUOTE = CorpusDoc(
    doc_id="quote", text="the canary in the coal mine rose early", doc_date="2024-02-01"
)
STALE = CorpusDoc(doc_id="stale", text="post cutoff text", doc_date="2024-05-02")
CORPUS = {d.doc_id: d for d in (LONG, WIDE, QUOTE, STALE)}


def reason(**overrides) -> dict:
    r = {
        "reason_id": "r1",
        "premise": "p",
        "mechanism": "m",
        "answer_implication": "a",
    }
    r.update(overrides)
    return r


def answer(*reasons, **extra) -> dict:
    out = {"task_id": "t", "entity_predictions": [], "submitted_reasons": list(reasons)}
    out.update(extra)
    return out


def codes(ans: dict, corpus: dict | None = None) -> list[str]:
    return sorted(
        f.code for f in check_submitted_reasons(ans, corpus or CORPUS, CUTOFF)
    )


def cite(doc_id: str, start: int, end: int) -> dict:
    return {"doc_id": doc_id, "span_start": start, "span_end": end}


# --------------------------------------------------------------------------- #
# absent / clean                                                               #
# --------------------------------------------------------------------------- #

def test_absent_field_is_clean():
    assert codes({"task_id": "t", "entity_predictions": []}) == []


def test_clean_reason_with_citation_passes():
    assert codes(answer(reason(citations=[cite("long", 0, 100)], scope={"entities": ["E"]}))) == []


# --------------------------------------------------------------------------- #
# shape                                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", [{}, "r1", None, 3])
def test_field_not_a_list(value):
    assert codes({"entity_predictions": [], "submitted_reasons": value}) == ["reasons_shape"]


def test_empty_list_flagged():
    assert codes(answer()) == ["reasons_shape"]


def test_four_reasons_flagged():
    assert codes(answer(*(reason(reason_id=f"r{i}", premise=f"p{i}") for i in range(4)))) == ["reasons_shape"]


def test_three_reasons_pass():
    assert codes(answer(*(reason(reason_id=f"r{i}", premise=f"p{i}") for i in range(3)))) == []


def test_reason_not_an_object():
    assert codes(answer("just text")) == ["reasons_shape"]


@pytest.mark.parametrize("field", ["reason_id", "premise", "mechanism", "answer_implication"])
def test_missing_required_field(field):
    r = reason()
    del r[field]
    assert codes(answer(r)) == ["reasons_shape"]


@pytest.mark.parametrize("field", ["reason_id", "premise", "mechanism", "answer_implication"])
def test_non_string_field(field):
    assert codes(answer(reason(**{field: 7}))) == ["reasons_shape"]


@pytest.mark.parametrize(
    "scope", [["E"], {"entities": "E"}, {"entities": [1]}]
)
def test_bad_scope(scope):
    assert codes(answer(reason(scope=scope))) == ["reasons_shape"]


@pytest.mark.parametrize(
    "citations",
    [
        {"doc_id": "long"},  # not a list
        ["long"],  # item not an object
        [{"doc_id": "long", "span_start": 0}],  # missing span_end
        [{"doc_id": 5, "span_start": 0, "span_end": 3}],  # non-string doc_id
        [{"doc_id": "long", "span_start": "0", "span_end": 3}],  # non-integer offset
        [{"doc_id": "long", "span_start": True, "span_end": 3}],  # bool is not an integer
        [{"doc_id": "long", "span_start": -1, "span_end": 3}],  # negative
    ],
)
def test_bad_citation_shape(citations):
    assert codes(answer(reason(citations=citations))) == ["reasons_shape"]


# --------------------------------------------------------------------------- #
# citation resolution (the judge never sees these; not a refusal)             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "c",
    [
        cite("nope", 0, 5),  # unknown doc
        cite("stale", 0, 5),  # post-cutoff doc
        cite("quote", 5, 5),  # empty range
        cite("quote", 0, 100),  # past document end
    ],
)
def test_unresolving_citation_flagged(c):
    assert codes(answer(reason(citations=[c]))) == ["reason_citation"]


# --------------------------------------------------------------------------- #
# caps                                                                         #
# --------------------------------------------------------------------------- #

def test_citation_at_8000_passes():
    assert codes(answer(reason(citations=[cite("long", 0, 8_000)]))) == []


def test_citation_at_8001_flagged():
    assert codes(answer(reason(citations=[cite("long", 0, 8_001)]))) == ["cap_citation_chars"]


from baselines.guardrails_example import citation_rail as rail  # noqa: E402


def _compact_len(items: list) -> int:
    s = json.dumps(items, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return len(s.encode("utf-8")) - 2


def _evidence_bytes(spans: list[tuple[str, int, int]], reason_id: str = "r1") -> int:
    return _compact_len([{"doc_id": d, "span_start": a, "span_end": b,
                          "trusted_text": CORPUS[d].text[a:b], "reason_id": reason_id} for d, a, b in spans])


def test_the_byte_caps_sum_to_the_judge_read_limit():
    assert (rail.MAX_ANSWER_BYTES, rail.MAX_REASON_BYTES, rail.MAX_EVIDENCE_BYTES) == (3_000, 6_500, 46_500)
    assert rail.MAX_JUDGE_BYTES == 56_000


def _evidence(extra: int) -> dict:
    spans = [("long", i * 8_000, (i + 1) * 8_000) for i in range(5)]  # 40,000 ASCII chars
    base = _evidence_bytes(spans + [("long", 40_000, 40_001)]) - 1
    last = 40_000 + (rail.MAX_EVIDENCE_BYTES - base) + extra
    spans.append(("long", 40_000, last))
    assert _evidence_bytes(spans) == rail.MAX_EVIDENCE_BYTES + extra
    return answer(reason(citations=[cite(d, a, b) for d, a, b in spans]))


def test_evidence_bytes_at_the_cap_pass():
    assert codes(_evidence(0)) == []


def test_evidence_bytes_one_past_the_cap_flagged():
    assert codes(_evidence(1)) == ["cap_evidence_bytes"]


def test_many_one_character_citations_count_their_json():
    # 1,000 characters cited, each citation's JSON counts: far over the evidence cap
    ans = answer(reason(citations=[cite("long", i, i + 1) for i in range(1_000)]))
    assert codes(ans) == ["cap_evidence_bytes"]


def test_two_byte_text_counts_in_bytes():
    ans = answer(reason(citations=[cite("wide", 0, 8_000) for _ in range(3)]))  # 24,000 chars, 48,000+ bytes
    assert codes(ans) == ["cap_evidence_bytes"]


def _reasons(extra: int) -> dict:
    r = reason(premise="", mechanism="m", answer_implication="a")
    used = _compact_len([{"reason_id": "r1", "premise": "", "mechanism": "m", "answer_implication": "a"}])
    r["premise"] = "p" * (rail.MAX_REASON_BYTES - used + extra)
    return answer(r)


def test_reason_bytes_at_the_cap_pass():
    assert codes(_reasons(0)) == []


def test_reason_bytes_one_past_the_cap_flagged():
    assert codes(_reasons(1)) == ["cap_reason_bytes"]


def test_six_thousand_plain_characters_of_reason_text_fit():
    ans = answer(*[reason(reason_id=f"r{n}", premise=str(n) * 2_000, mechanism="", answer_implication="")
                   for n in (1, 2, 3)])
    assert codes(ans) == []


def test_answer_bytes_over_the_cap_flagged():
    ans = answer(reason())
    ans["entity_predictions"] = [{"entity_id": "E", "point_forecast": int("9" * 3_100), "claims": []}]
    assert codes(ans) == ["cap_answer_bytes"]


def test_unresolving_citations_do_not_count_toward_bytes():
    ans = _evidence(0)
    ans["submitted_reasons"][0]["citations"].append(cite("stale", 0, 5))
    assert codes(ans) == ["reason_citation"]


def test_uris_in_cited_text_count_as_masked():
    """The grader masks a URI in cited text with the same number of U+2588 (3 bytes each)."""
    uri = CorpusDoc(doc_id="uri", text="see https://example.org/ " + "x" * 7_000, doc_date="2024-02-01")
    corpus = {**CORPUS, "uri": uri}

    def raw_bytes(spans):
        return _compact_len([{"doc_id": d, "span_start": a, "span_end": b,
                              "trusted_text": corpus[d].text[a:b], "reason_id": "r1"} for d, a, b in spans])

    head = [("long", i * 8_000, (i + 1) * 8_000) for i in range(4)] + [("uri", 0, len(uri.text))]
    base = raw_bytes(head + [("long", 32_000, 32_001)]) - 1
    spans = head + [("long", 32_000, 32_000 + (rail.MAX_EVIDENCE_BYTES - 10 - base))]
    assert raw_bytes(spans) == rail.MAX_EVIDENCE_BYTES - 10  # under the cap as raw text
    ans = answer(reason(citations=[cite(d, a, b) for d, a, b in spans]))
    assert codes(ans, corpus) == ["cap_evidence_bytes"]  # 20 masked characters add 40 bytes


# --------------------------------------------------------------------------- #
# deny list                                                                    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("phrase", ["https://example.org", "see /home/x", "OUTCOME.JSON", "Team Name"])
def test_deny_list_in_mechanism_flagged(phrase):
    assert codes(answer(reason(mechanism=f"because {phrase}"))) == ["deny_list"]


def test_deny_list_in_answer_implication_flagged():
    assert codes(answer(reason(answer_implication="beats the leaderboard"))) == ["deny_list"]


def test_whole_verbatim_premise_quote_is_exempt():
    assert codes(answer(reason(premise="  canary in the coal mine\n"))) == []


def test_premise_that_adds_a_word_is_checked():
    assert codes(answer(reason(premise="canary in the coal mine indeed"))) == ["deny_list"]


def test_verbatim_quote_does_not_exempt_mechanism():
    assert codes(
        answer(reason(premise="canary in the coal mine", mechanism="canary in the coal mine"))
    ) == ["deny_list"]


# --------------------------------------------------------------------------- #
# the worked example in the docs resolves against the shipped exemplar corpus #
# --------------------------------------------------------------------------- #

WORKED_EXAMPLE_DOCS = ["SUBMISSION_CLI.md", "docs/AUTHORING-GUIDE.md"]


def _worked_examples(rel: str) -> list[dict]:
    text = (REPO / rel).read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(.*?)```", text, flags=re.S)
    return [json.loads(b) for b in blocks if '"submitted_reasons"' in b]


@pytest.mark.parametrize("rel", WORKED_EXAMPLE_DOCS)
def test_documented_worked_example_is_clean(rel):
    examples = _worked_examples(rel)
    assert len(examples) == 1, f"{rel}: expected one worked submitted_reasons example"
    unit = REPO / "units" / "t4-EXAMPLE-eps-beat"
    task = json.loads((unit / "task.json").read_text(encoding="utf-8"))
    corpus = load_corpus(unit / "corpus")
    ex = examples[0]
    assert check_submitted_reasons(ex, corpus, task["cutoff_date"]) == []
    # Every premise in the example is a verbatim quote of the first passage it cites,
    # so the offsets printed in the docs are the offsets of the quoted text.
    for r in ex["submitted_reasons"]:
        c = r["citations"][0]
        assert corpus[c["doc_id"]].text[c["span_start"]:c["span_end"]] == r["premise"]

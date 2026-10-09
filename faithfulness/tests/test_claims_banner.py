"""Scorer 5.3.0: the local check's banner counts the claims the scorer checks.

Only the first 20 claims about each entity are checked and counted; the banner says how many
claims it scores (the counted ones) and, when the cap applies, how many it ignores.
"""

from __future__ import annotations

from typing import Any

from faithfulness.judge import claims_banner


def _answer(per_entity: dict[str, int]) -> dict[str, Any]:
    return {
        "entity_predictions": [
            {
                "entity_id": entity,
                "claims": [
                    {"doc_id": "d", "span_start": 0, "span_end": 1, "claim": f"c{i}"}
                    for i in range(n)
                ],
            }
            for entity, n in per_entity.items()
        ]
    }


def test_the_banner_counts_only_checked_claims() -> None:
    line = claims_banner(_answer({"A": 25, "B": 3}), 2, "answer.json", "t4-SYNTH")
    assert line.startswith("Scoring 23 claim(s) for 2 roster entit(y/ies)")
    assert line.endswith("; 5 claim(s) after the first 20 about an entity are ignored")


def test_the_banner_is_unchanged_without_ignored_claims() -> None:
    line = claims_banner(_answer({"A": 20, "B": 1}), 2, "answer.json", "t4-SYNTH")
    assert line == (
        "Scoring 21 claim(s) for 2 roster entit(y/ies) from 'answer.json' against unit "
        "'t4-SYNTH'"
    )

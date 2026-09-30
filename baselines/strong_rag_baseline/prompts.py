"""Prompt construction for the strong RAG baseline.

One prompt per entity. The model sees the entity's tabular features and the
top-K retrieved excerpts, each tagged with its ``doc_id``, and must return a
single JSON object. Evidence quotes are required to be verbatim substrings of
the provided excerpts — the span finder maps them back to exact character
offsets, and anything it cannot locate is dropped rather than cited loosely.
"""
from __future__ import annotations

import json

from .indexer import Chunk

SYSTEM_PROMPT = """\
You are a careful financial analyst. You predict a target for one entity using ONLY the
evidence excerpts provided — no outside knowledge about events after the stated cutoff date.
You must respond with a single JSON object and nothing else. Every evidence quote you return
must be copied verbatim, character for character, from one of the provided excerpts."""

_TARGET_INSTRUCTIONS = {
    "classification": (
        'Set "label" to exactly one of the allowed labels. '
        'Set "point_forecast" to your best numeric estimate of the underlying quantity '
        "if one is defined for this task, else null."
    ),
    "regression": (
        'Set "point_forecast" to your numeric prediction of the target. '
        '"label" may be null.'
    ),
    "ranking": (
        'Set "point_forecast" to the predicted metric value; entities are ranked by it. '
        '"label" may be null.'
    ),
}


def build_user_prompt(
    task: dict, entity: dict, retrieved: list[Chunk]
) -> str:
    target = task.get("target", {})
    target_type = target.get("type", "classification")
    lines: list[str] = []

    lines.append(f"TASK: {task.get('prompt', '')}")
    lines.append(f"CUTOFF DATE: {task.get('cutoff_date', '')}")
    lines.append(f"TARGET: {target.get('name', '')} ({target_type})")
    if target.get("labels"):
        lines.append(f"ALLOWED LABELS: {', '.join(target['labels'])}")
    interval_level = task.get("interval_level", 0.90)

    lines.append("\nENTITY:")
    for key, value in entity.items():
        if key in ("corpus_ref",):
            continue
        lines.append(f"  {key}: {value}")

    lines.append("\nEVIDENCE EXCERPTS (cite only these):")
    for i, chunk in enumerate(retrieved, 1):
        lines.append(f"[{i}] doc_id={chunk.doc_id} (doc_date={chunk.doc_date})")
        lines.append(f'"""{chunk.text}"""')

    schema = {
        "label": "string or null",
        "point_forecast": "number or null",
        "interval": {
            "level": interval_level,
            "lo": "number",
            "hi": "number",
        },
        "evidence": [
            {
                "doc_id": "doc_id of the excerpt the quote comes from",
                "quote": "verbatim substring copied from that excerpt",
                "claim": "one factual sentence the quote directly supports",
            }
        ],
    }
    lines.append(
        "\nRespond with ONE JSON object of this shape (no markdown fences, no prose):"
    )
    lines.append(json.dumps(schema, indent=2))
    lines.append(f"\n{_TARGET_INSTRUCTIONS.get(target_type, _TARGET_INSTRUCTIONS['classification'])}")
    lines.append(
        f'The "interval" must be your {int(interval_level * 100)}% prediction interval for the '
        "numeric target: wide enough that you expect the realized value to fall inside it "
        f"{int(interval_level * 100)}% of the time, and no wider. "
        "Give 2 to 4 evidence entries. Each claim must be fully supported by its quote alone."
    )
    return "\n".join(lines)

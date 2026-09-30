# Guardrails example — a local citation rail for Track 4 agents

## Executive summary (read this first)

A small, optional, participant-side rail you can run on your OWN draft answer before
submitting. It performs two checks: every cited document must exist in the frozen corpus with
`doc_date <= cutoff_date`, and every claim must carry a well-formed `(doc_id, span_start,
span_end)` that resolves inside the cited document's text. These are exactly the mistakes the
stale-filing adversarial variants are built to elicit, so catching them locally reduces YOUR
refused units and false claims. **The rail is advisory.** The competition's embargo and
faithfulness checks are
deterministic, organizer-side code and are the authority on every submission — passing this
rail guarantees nothing about scoring; it only stops you from submitting a claim that would
certainly fail.

A third check, `check_submitted_reasons`, covers the optional `submitted_reasons` field the
reasoning grader reads: its shape, whether each citation resolves before the cutoff, the
published caps, the deny list and content-duplicate reasons (see "Checking your reasons" below).

A fourth, `check_claim_rules`, runs the scorer 5.2.1 **deterministic claim rules** on a whole
answer with the scorer's own code (see "Checking your claims (scorer 5.2.0)" below). It is the
only part of this rail that imports this repository's `qfbench2_track_analysis`; the other three
are pure standard library.

## What's here

| File | What |
|---|---|
| `citation_rail.py` | The checks. `load_corpus`, `filter_retrieved` (retrieval-time date rail), `check_answer` (submission-time rail → findings list; pass `task=` to check `"task"` citations), `check_submitted_reasons` (the reasoning field's shape, caps, deny list and duplicates), `check_claim_rules` (the scorer's deterministic 5.2.1 claim rules). |
| `demo.py` | Offline demo on `units/t4-EXAMPLE-eps-beat`: an agent "accidentally" cites a live-fetched post-cutoff snippet and emits one malformed span; the rail flags both, the clean claim passes. It also checks three submitted reasons: two clean, one with a URL in its mechanism, which the deny list flags. |
| `rails/` | Illustrative [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) wiring of the same checks as an output rail (`config.yml`, `flows.co`, `actions.py`). Requires `pip install nemoguardrails`; the demo does not. |
| `tests/` | Unit tests for the rail checks. |

## Run the demo

```bash
# from the repo root
python -m baselines.guardrails_example.demo
```

Expected output ends with `DEMO PASS — the rail caught exactly the planted problems.` The
planted post-cutoff document is synthetic text invented for the demo, marked as such in the
source.

## Using the rail in your own agent

```python
from baselines.guardrails_example.citation_rail import (
    load_corpus, filter_retrieved, check_answer,
)

corpus = load_corpus(unit_dir / "corpus")
usable, stale = filter_retrieved(list(corpus.values()), task["cutoff_date"])
# ... retrieve/reason over `usable` only ...
findings = check_answer(draft_answer, corpus, task["cutoff_date"], task=task)
if findings:
    ...  # drop or repair the flagged claims, then redraft
```

Two placements, use both: filter the retrieval pool up front so stale material never reaches
the reasoning step, and re-check the assembled answer just before writing it out.

A claim may cite the task table with `"doc_id": "task"`. Pass `task=` (the parsed `task.json`)
and the span is checked against the task-table text the scorer builds
(`qfbench2_track_analysis.corpus.task_table_text`): outside the table is `bad_span`, outside the
citing entity's own row is `task_row` (the scorer counts that claim false, as a wrong-entity
citation). Without `task=` such a citation is reported `task_unchecked`, never `unknown_doc`.

## Checking your claims (scorer 5.2.0)

```python
from baselines.guardrails_example.citation_rail import check_claim_rules

for finding in check_claim_rules(draft_answer, unit_dir):
    print(finding)  # e.g. [claim_unanchored] entity=AAPL claim#3: states a figure no cited span carries ...
```

Run it from the repository root (it imports `qfbench2_track_analysis`). It hydrates the unit the
way the scorer does and calls the scorer's own `evaluate_claims`, so its verdicts are the
scorer's; the organizer runs a parity test of the two over real answers. Each finding is one
reason a claim is false, and a false claim costs its share of the unit:

| code | what it means |
|---|---|
| `claim_wrong_entity` | a citation of a document the unit manifest does not label for the claim's entity (nor mark `shared`), or a `"task"` span outside the entity's own row |
| `claim_out_of_range` | offsets that are not a slice of the cited document |
| `claim_malformed` | empty, over 4,000 characters, or over 400 judge tokens |
| `claim_unanchored` | a figure no cited span carries (every figure, whole span; a word-for-word quote of a cited span passes at any span length; your scored values and the unit's own entity names and tickers are exempt; a span over 8,000 characters anchors no figure) |
| `unit_refused` | the scorer refuses the whole unit before any claim rule (a missing, extra or repeated entity, a non-finite value, an unresolved, undated or post-cutoff citation); run `check_answer` to see which citation |
| `claim_tokens_unchecked` | needs tokenizer: the judge's tokenizer is not installed here, so the 400-token cap was not checked. Install `transformers` and the judge models (nothing is downloaded by this check), or pass `token_counter=` |

## Checking your reasons

```python
from baselines.guardrails_example.citation_rail import check_submitted_reasons

for finding in check_submitted_reasons(draft_answer, corpus, task["cutoff_date"]):
    print(finding)  # e.g. [cap_citation_chars] submitted_reasons reason#0: ...
```

An absent `submitted_reasons` field is clean: leaving reasons out never costs anything. A
`submitted_reasons` block that does not match the schema (an empty list, more than 3 reasons, or
a reason missing a required field) makes the whole answer invalid, like any schema error, and the
unit takes the worst value; that is the `reasons_shape` finding below, so clear it before you
submit. The finding codes:

| code | what it means | what the grader does |
|---|---|---|
| `reasons_shape` | not a list of 1 to 3 objects; a required field (`reason_id`, `premise`, `mechanism`, `answer_implication`) missing or not a string; `scope`, `scope.entities` or `citations` of the wrong shape | the whole answer fails the schema; the unit takes the worst value |
| `reason_citation` | a citation to an unknown document, an empty or out-of-range span, or a document dated after the cutoff | the judge never sees that passage |
| `cap_citation_chars` | one citation spans more than 8,000 characters | the unit's reasoning is not judged (0) |
| `cap_answer_bytes` | your per-entity answer, as compact JSON, exceeds 3,000 UTF-8 bytes | the unit's reasoning is not judged (0) |
| `cap_reason_bytes` | your reasons (`reason_id`, `premise`, `mechanism`, `answer_implication`), as compact JSON, exceed 6,500 UTF-8 bytes | the unit's reasoning is not judged (0) |
| `cap_evidence_bytes` | the resolved cited passages with their doc id and offsets, as compact JSON (URIs masked), exceed 46,500 UTF-8 bytes | the unit's reasoning is not judged (0) |
| `deny_list` | a deny-list phrase in `mechanism` or `answer_implication`, or in a `premise` that is not a verbatim corpus quote of at least 3 words (URLs masked on both sides; a bare URL, `units/`, `canary` or a two-word quote is not one) | the grader refuses that unit's reasoning (0); the analysis score is unaffected |
| `duplicate_reason` | the same `premise`, `mechanism` and `answer_implication` as an earlier reason (case, whitespace and invisible characters ignored) | that reason is not judged and covers no target reason; the others are judged |

A `"task"` citation in a reason is a `reason_citation`: the grader resolves reason citations
against the corpus only, so the judge never sees it. `reason_id` values are not checked: the
grader renumbers reasons by position and does not require unique ids.

The byte counts follow the grader (reasons renumbered r1..r3, URIs in cited text masked); the per-entity answer is counted with every answer field present, so it can only over-count.
The contract itself, with a worked example on the exemplar unit, is in
[`SUBMISSION_CLI.md`](../../SUBMISSION_CLI.md#how-reasoning-is-scored).

## What this rail does, and does NOT do

It does: the date and shape rails on every claim citation (including `"task"` citations with
`task=`); every deterministic 5.2.0 claim rule (`check_claim_rules`: wrong entity, out of
range, malformed including the 400-judge-token cap when the tokenizer is installed, every figure
with the verbatim-quote pass, the name and ticker exemption, the 8,000-character span cap); and
the reasoning field's shape, citations, caps, deny list and duplicate reasons.

It does NOT:

- run NLI. The contradiction check (the ensemble's three-way P(contradiction) above 0.9 makes
  a claim false) needs the models; `faithfulness/judge.py --answer … --unit …` runs it.
- It does not judge your reasons. A reason can pass every check and still score 0 against
  the unit's target reasons.
- It does not validate the full answer schema (use `templates/answer.example.json` and
  `qfbench2_common/schemas/analysis.schema.json` for that).
- It is not part of scoring or admissibility, and never will be — organizer-side gates are
  computed independently of anything you run locally.

# Track 4 — How a Unit Is Built (Participant Guide)

## Executive summary (read this first)

A Track 4 **unit** is a self-contained directory: a task file, a task card, a checksum manifest,
and a frozen corpus of documents. This page walks through that layout field by field and says, for
each one, **who reads it** — your agent, the embargo gate, the faithfulness judge, the scorer, or
nobody. Knowing which fields are load-bearing is what stops you building a retrieval pipeline
around a provenance label the scorer never looks at, or missing the one field that decides whether
a citation resolves at all. Everything here is read off the shipped units and the shipped scorer,
so you can check any claim on this page against
`units/t4-EXAMPLE-eps-beat/` and `qfbench2_track_analysis/scoring.py`.

This is a guide to *reading* a unit, not to writing one. Task authoring is an organiser process and
its material — resolved outcomes, adversarial variant construction, generation and review logs — is
not published, by design: it is the answer key. If you are looking for what to predict, start with
[`CATEGORIES.md`](CATEGORIES.md); if a term is unfamiliar, start with [`CONCEPTS.md`](CONCEPTS.md).

---

## The directory

```
units/t4-EXAMPLE-eps-beat/
  card.toml       # the unit's parameters: gates, thresholds, environment, embargo
  task.json       # what to predict, for which entities, from what evidence
  manifest.json   # per-file checksum manifest for the whole unit
  corpus/         # the frozen evidence, one JSON document per file
```

Your container does not see this path. The harness mounts the **unit directory itself** at
`/input`, so your agent reads `/input/task.json` and `/input/corpus/` — see
[`../SUBMISSION_CLI.md`](../SUBMISSION_CLI.md) for the full invocation.

---

## `task.json` — the participant-facing task

The exemplar is `units/t4-EXAMPLE-eps-beat/task.json`. The fields that matter:

| Field | Read by | What it is |
|---|---|---|
| `task_id` | the scorer | Copy it verbatim into `answer.json`. A mismatch is a whole-submission failure. |
| `schema_version` | the scorer | The answer-schema generation this unit speaks. |
| `family` | nobody | A descriptive slug. No enum is enforced; do not switch on it. |
| `target` | **you and the scorer** | The nested block `{name, type, labels}`. `type` is one of `classification`, `regression`, `ranking`. |
| `prompt` | **you** | The plain-English statement of what to predict, including the units the number must be in. |
| `cutoff_date` | **the embargo gate** | Cited evidence must be dated on or before this. |
| `resolution_date` | context | When the outcome became known. The gap to `cutoff_date` is the horizon; it varies widely across units. |
| `interval_level` | the scorer | The level your `interval.level` must equal. |
| `corpus_manifest` | tooling | Relative path to the unit's manifest. |
| `faithfulness_rubric` | nobody (legacy) | Prose written for the admission gate used before 5.2.0 (NLI score above 0.5 per claim, 80% of claims supported). Neither scorer 5.2.0 nor the reasoning grader reads it; ignore it. The rules that apply are in `SUBMISSION_CLI.md` ("How faithfulness is scored") and `docs/CONCEPTS.md`. |
| `entities` | **you and the scorer** | The table. One object per row; `entity_id` is the key the scorer aligns on. |
| `notes` | nobody | Free text. |

**The target block is nested.** It is `target.type`, not a flat `target_type` key, in `task.json`.
The flat spelling belongs to two other places: the `answer.json` you emit, and `[scoring.params]`
in `card.toml`. The shipped baseline reads `task.get("target", {}).get("type")`, and no reader in
this repository consumes a flat `target_type` out of `task.json`.

**Entity columns are pre-cutoff facts.** Feature columns differ per family — `consensus_eps`,
`prior_year_q_eps`, `start_yield_pct`, `latest_precutoff_estimate`, `open_interest_20241022`. None
of them is the answer; several are the answer's value one period earlier, which is the naive
baseline a regression unit measures your skill against.

---

## `card.toml` — the unit's parameters

`card.toml` is authoritative for everything the scorer is configured with. The blocks you should
read before designing an agent:

- `[scoring.params]` — `faithfulness_threshold` (retired in 5.2.0: only the legacy 0.80 is
  accepted, and it means "use the per-claim penalty"; `penalty_k` and `contradiction_bar` are
  fixed scorer constants and a card naming them is refused), `interval_level`, `target_type`,
  `composite_weights`, `tau_citation`,
  and the pinned NLI ensemble. These are the numbers the
  gate and the composite actually use.
- `[environment]` — the compute grant and network mode the harness enforces.
- `[agent] timeout_sec` — the per-unit wall clock. Exceeding it is a `g2` timeout failure.
- `[embargo]` — which field on the task carries the cutoff (`cutoff_date`), which field on each
  corpus document carries its date (`doc_date`), and that the check is strict.
- `[contamination] canary_guid` — a per-unit contamination marker. Never emit it.

Prose in this repository is not allowed to hand-restate these values: the guards in
`baselines/tests/test_docs_match_artifacts.py` fail CI when a document quotes a threshold or an
`[environment]` grant the exemplar card does not carry. Read the card.

---

## `manifest.json` — the checksum manifest

The manifest at the **unit root** is a per-file checksum manifest: a `files[]` array whose entries
each carry a `path`, a real `sha256`, a `bytes` count, and provenance fields (`role`, `source`,
`license`, `split`, `redistributable`, `pii_stripped`). `verify_manifest` — run by the
`validate-units` CI job and by the harness before mounting — checks it **exactly, in both
directions**: every declared entry must exist and hash to the recorded digest, and every regular
file inside the subtrees the manifest covers must be declared. Which subtrees those are varies:
ten of the eleven published units declare `card.toml` and `task.json` alongside their corpus files,
while the exemplar declares its corpus only.

It is **not** a corpus index. Per-document metadata (`doc_date`, `form_type`, ticker) lives in each
corpus document, where the scorer reads it, and in `task.json`. A `documents[]`-style index in a
file named `manifest.json` fails verification.

Ten of the eleven published units also carry a **second** manifest inside `corpus/`, covering the
corpus files alone; the exemplar does not. Resolve citations through the document files themselves
rather than assuming one manifest layout.

---

## `corpus/` — the frozen evidence

One JSON document per file, at `corpus/<doc_id>.json`. Two fields are load-bearing:

| Field | Read by | Why it matters |
|---|---|---|
| `text` | **the faithfulness judge** | The document's plain text. This is the premise every entailment check is run against. |
| `doc_date` | **the embargo gate** | ISO-8601. A citation into a document dated after `cutoff_date` fails the unit. |

`doc_id` must equal the filename stem: the scorer resolves a citation by looking up
`corpus/<doc_id>.json`. Nothing parses the id's internal structure — the shipped convention is
`EDGAR_{cik}_{form}_{YYYYMMDD}` for filings and `{SOURCE}_{SERIES}_{YYYYMMDD}` for data snapshots,
but uniqueness and the filename match are the only real constraints.

The remaining fields — `source`, `form_type`, `ticker`, `cik`, `title`, `note`, `pii_stripped` —
are provenance records. No scoring code reads them. They are useful to you for filtering and
retrieval; do not build a contract on them.

**Where the text lives matters.** `qfbench2_common.scoring.faithfulness._doc_text` accepts a raw
string, a dict with a flat `text` string, or a dict with a `spans` list of `{text: ...}` objects,
which it joins with spaces. Content anywhere else resolves to the empty string, every citation into
that document yields an empty premise, and faithfulness scores zero. Treat the flat `text` field as
the normal case.

---

## Character offsets: how a citation points at a passage

A citation is `{doc_id, span_start, span_end, claim}`. The offsets are zero-indexed character
positions into the resolved document text — the same string `_doc_text` returns — so
`text[span_start:span_end]` is the premise the judge is shown. Compute them against the document
you actually loaded, and verify by slicing before you emit.

Your `claim` string **is** what the judge reads: the premise is the passage your offsets select
and the hypothesis is your claim. From scorer 5.2.0 faithfulness is a **per-claim penalty**: each
claim is either false or neutral, and the unit's score is multiplied by
`1 - F / (F + min(T, 3 × E))` (F false claims, T other claims, E entities): each false claim
costs a share of the unit, other claims beyond 3 × E in total do not dilute that cost, and a unit with no
false claims is not penalised. A claim is false when it cites a
document that is not about its entity, cites offsets outside the document, is empty, over 4000
characters or over 400 judge tokens, states **any** figure that no span it cites carries (read
against the whole cited span; a span over 8,000 characters anchors no figure; dates, periods,
counts of periods and identifiers are not figures; numbers inside the unit's own entity names or
tickers are exempt, and so is a figure that equals a scored value you submitted: your
point forecast on a regression or ranking unit, and your interval bounds only when the unit's
interval leg is scored; the passages' scale steps apply but no rounding, a figure whose written
direction contradicts the value's sign is not exempt, and your rank is never exempt), or when the judge finds the passage **contradicts** it
(three-way contradiction probability above `contradiction_bar` = 0.9). A word-for-word quote of a
span it cites passes the figure check and is not put to the judge; a verbatim quote passes even when the span it cites is over 8,000 characters (the quote is
looked for in the first 200,000 characters of the span); the 8,000-character cap applies to
every other claim. Everything else is neutral: never charged, and it earns nothing here.

**Claims are extractive facts.** Write claims that say what the passage says, with the figures it
carries: **every figure in a claim must appear in a passage it cites**. A figure you computed (a
change between two rows, a ratio, a sum) is not in the passage; put the computation in
`submitted_reasons` (the `mechanism` field), where derivations are judged, and keep the claim to
the figures it cites. A value the task gives you (a row of `task.json` `entities`) is cited with
`"doc_id": "task"` and a span inside that entity's row of the task table
(`qfbench2_track_analysis.corpus.task_table_text`; see `CONCEPTS.md`, "Citing the task table").

Every document you cite must also be *about the entity you cite it for*: the unit manifest labels
each corpus document with its roster entities (`entity_ids`) or as market-wide (`shared`), and a
claim citing another entity's document is false (before 5.2.0 it refused the whole unit). The
check catches fabricated or misattributed evidence; whether the evidence supports your forecast is
reasoning grading's question. `penalty_k` = 1 and `contradiction_bar` = 0.9 are fixed scorer
constants (a card or plan that names either is refused); cards still carry the retired `faithfulness_threshold` = 0.80, which 5.2.0 reads only as "use the
per-claim penalty" (see `CONCEPTS.md`).

---

## What "the outcome" is, and where it is not

Each unit resolves against a per-entity outcome roster the organizers hold. It is not published,
and no field of it appears in this repository. Two properties of it are worth knowing
because they shape what a well-formed submission looks like:

- The roster covers the entity roster **exactly** — the same `entity_id` set, no more and no less.
  Your `entity_predictions[]` must do the same.
- Where a family has a numeric target, **every** row carries it or none does. There is no unit in
  which some rows are scored numerically and others are not. A unit with no numeric target is a
  pure-label unit, and its composite drops the calibration leg entirely.

---

## How reasoning is scored

Track 4 has a second grader beside the analysis score and the faithfulness penalty: an LLM judge
panel that grades your **reasons**. Your reasons go in one optional top-level field of
`answer.json`, `submitted_reasons`, next to `entity_predictions`. The reasoning grader reads
nothing else you write: `claims`, `evidence_trace` and `notes` are not reasons. An answer
without the field has submitted no reasons.

**The field.** `submitted_reasons` is a list of 1 to 3 reasons. Omit the field to submit none.
A `submitted_reasons` block that does not match the schema (an empty list, more than 3
reasons, or a reason missing a required field) makes the whole answer invalid, like any schema
error, and the unit takes the worst value; run the local checker (`check_submitted_reasons`)
first. Each reason is an object:

| field | required | what it holds |
|---|---|---|
| `reason_id` | yes | a string you choose; the grader does not read it for grading (it renumbers your reasons r1, r2, r3 by position) and does not check that ids are unique, but unique ids keep your reasons apart |
| `premise` | yes | the evidence-grounded fact |
| `mechanism` | yes | why that fact moves the answer |
| `answer_implication` | yes | what it implies for your submitted answer, naming the entities |
| `scope` | no | an object with `entities`: a list of `entity_id` strings |
| `citations` | no | a list of `{doc_id, span_start, span_end}` (integers >= 0), in the same character-offset convention as `claims` (see [Character offsets](#character-offsets-how-a-citation-points-at-a-passage)) |

A citation must resolve in the frozen corpus and its document must be dated on or before the
cutoff; otherwise the judge never sees that passage. The task-table citation `"doc_id": "task"`
is for claims only: the grader resolves reason citations against the corpus alone, so a
`"task"` citation in a reason resolves to nothing and the judge never sees it. The judge reads
the task statement and each entity's id and name, not the rows of the task table: state a task
value you rely on in the premise; the rest of the reason is judged as usual.

**Duplicate reasons.** A reason whose `premise`, `mechanism` and `answer_implication` equal an
earlier reason's (compared after Unicode NFC normalisation, with invisible format characters
removed, case folded and whitespace runs collapsed) is not sent to the judge, so it covers no
target reason. A different `reason_id` does not make it a new reason.

The schema is `analysis.schema.json` in
the shared toolkit (`qfbench2_common/schemas/`); the submission side is in
[`SUBMISSION_CLI.md`](../SUBMISSION_CLI.md#how-reasoning-is-scored).

**What the judge sees, and what it grades.** The judge reads the task statement and entity
list; your per-entity answer (only the fields the unit declares, of `label`, `point_forecast`,
`interval` and `label_probs`, taken from your `entity_predictions`); each reason's `premise`,
`mechanism` and `answer_implication`; and the corpus text your citations resolve to. It does
not see `scope` or the raw citations. It compares your reasons with the unit's hidden target
reasons (part of the answer key, and like the outcome not published) and grades four components, `target_reason_coverage`, `evidence_grounding`,
`inferential_link` and `answer_consistency`, plus the flags `valid_grounded_premise`,
`has_answer_implication` and `contradiction`, with 5 judge votes per cell. A target reason that
none of yours covers scores 0 against a denominator of all the unit's target reasons, so
submitting fewer reasons never scores higher. From Track 4 scorer 5.2.0 the reasoning score is
a **bonus** on top of the analysis score (final-score/v2):

    final = -0.27 + 1.27 x analysis + 0.25 x reasoning

`analysis` is your 0..1 analysis score after the per-claim faithfulness penalty, shown on the
old leaderboard scale (`-0.27 + 1.27 x analysis`: 0 shows -0.27, the old worst case, and 1 shows
1.0); `reasoning` is in [0, 1]. The bonus is uncapped, so the maximum is 1.25. A keyed unit with
no judged reasons (missing, not judged, or refused for a cap or the deny list) adds 0 to the
bonus: leaving reasons out never costs anything. A block that fails the schema is different (see
"The field" above). Reasoning is graded offline after the Final, on the held-out units, and never
appears on a CodaBench board, the Development leaderboard included; the Development leaderboard
shows the analysis score only. The reasons format is the same everywhere, so practise it on the
Development practice units.

**Old scores and resubmitting.** Leaderboard scores already posted under the earlier scorer stay
as they were (frozen, not re-scored). A submission made with the new starter package is scored
with scorer 5.2.0 and this final formula.

**Your answer rows.** The judge's per-entity answer is built from `entity_predictions` in the
same `answer.json`: each row keeps `entity_id` and the answer fields the unit declares, and every
other field (`claims`, `rank`, and any of `label`, `point_forecast`, `interval` or
`label_probs` the unit does not declare) is dropped before the judge sees it. The rows must name
every entity of the task's entity list exactly once. Their order does not matter: they are put in
entity-list order. A missing, extra or repeated entity, or a row without a declared field, means
that unit's reasoning is not judged and scores 0; give every row `label` (classification) or
`point_forecast` (regression, ranking) besides the required `interval`. A top-level
`submitted_answer` is not read.

**Which fields a unit declares.** The declaration is part of the unit's reasoning key, which is
organizer material and not in the unit you receive. In the released units: regression and
ranking units declare `point_forecast` and `interval`; classification units declare `label`,
plus `interval` on units that score an interval leg (numeric truth, and `interval_leg` not set
to false in `card.toml`). No unit declares `label_probs`. An undeclared field is dropped before
the judge reads your answer; it is not an error and costs nothing.

**Caps.** Over any cap, that unit's reasoning is not judged and scores 0. Nothing is clipped.

| cap | limit, per unit |
|---|---|
| one citation: `span_end - span_start` | 8,000 characters (cite the passage, not the document) |
| your per-entity answer as the judge reads it (each row's `entity_id` and declared fields) | 3,000 bytes |
| your reasons as the judge reads them (`reason_id`, `premise`, `mechanism`, `answer_implication`) | 6,500 bytes |
| the cited passages as the judge reads them (each resolved citation's text with its `doc_id`, offsets and reason id) | 46,500 bytes |

The last three are counted the way the grader counts what the judge reads: UTF-8 bytes of compact
JSON. Plain ASCII text is one byte per character; a line break, quote or backslash is two (it is
escaped); accented letters, typographic quotes and non-Latin scripts take two to four; a control
character six; a URI in cited text is masked with the same number of `█` (three bytes each); and
every citation adds about 75 bytes of JSON around its text plus its `doc_id` and offsets. In practice: about 6,000 characters of
plain reason text over three reasons, and about 45,000 characters of plain cited text in a few
citations. The three caps add up to the grader's 56,000-byte limit on what the judge reads from
you, so an answer within them never reaches that limit. The local checker below reports each
(`cap_answer_bytes`, `cap_reason_bytes`, `cap_evidence_bytes`).

**Deny list.** The grader refuses a unit's request, and that unit's reasoning scores 0, if the
text you wrote contains any of these, case-insensitively, as a substring: `leaderboard`,
`canary`, `://`, `/home/`, `units/`, `reference/`, `outcome.json`, `team_id`, `team name`,
`participant_id`, `participant name`, `submission_id`, `other submission`. `mechanism` and
`answer_implication` are always checked. Exempt: the corpus text your citations resolve to,
and a `premise` that is, as a whole (surrounding whitespace aside), a verbatim quote of a
corpus document; a premise that adds any word of your own is checked. So do not put URLs or
file paths in your own words.

**Organiser faults.** If the grader fails on an organiser input (the task, the key, the
corpus, the judge forms or the policy), the grading run stops, the organiser fixes it and the
submission is re-graded. A unit that can never be graded is dropped from the reasoning score
for every submission, never for one submission only.

**Check it locally.** `check_submitted_reasons(answer, corpus, cutoff_date)` in
`baselines/guardrails_example/citation_rail.py` (standard library only, advisory) flags the
shape errors, citations the judge would not see, each cap including the byte backstop, and
deny-list hits. The demo runs it: `python -m baselines.guardrails_example.demo`.

**Worked example** on the public dev unit `t4-EXAMPLE-eps-beat`. The offsets are real: each
citation slices exactly the quoted premise out of the document's flat text, so both premises
are verbatim quotes. The label and the reasoning are illustrative, not a statement about the
outcome.

```json
{
  "task_id": "t4-EXAMPLE-eps-beat",
  "entity_predictions": [
    {
      "entity_id": "AAPL",
      "label": "beat",
      "interval": {"level": 0.90, "lo": 1.42, "hi": 1.68},
      "claims": [
        {
          "doc_id": "EDGAR_0000320193_10Q_20240202",
          "span_start": 295,
          "span_end": 433,
          "claim": "Services net sales were $23.1 billion in the December quarter, up 11.3% year over year."
        }
      ]
    }
  ],
  "submitted_reasons": [
    {
      "reason_id": "r1",
      "premise": "total revenue is expected to grow low- to mid-single digits year over year; Services revenue is expected to grow double digits year over year; gross margin is expected to be between 46.0 and 47.0 percent",
      "mechanism": "Guidance of revenue growth at a steady 46 to 47 percent gross margin means gross profit, and with it earnings per share, should rise year over year in the March quarter rather than fall.",
      "answer_implication": "Supports a label of beat for AAPL: earnings growth of that kind puts diluted EPS above the 1.50 consensus.",
      "scope": {"entities": ["AAPL"]},
      "citations": [
        {"doc_id": "EDGAR_0000320193_8K_20240201", "span_start": 711, "span_end": 914}
      ]
    },
    {
      "reason_id": "r2",
      "premise": "Services net sales were $23.1 billion for the three months ended December 30, 2023, an increase of $2.3 billion, or 11.3%, year over year.",
      "mechanism": "Services carry a gross margin far above the company average, so double-digit Services growth lifts profit faster than revenue.",
      "answer_implication": "Adds to the case that AAPL's earnings clear the consensus by more than the 5% threshold (beat).",
      "scope": {"entities": ["AAPL"]},
      "citations": [
        {"doc_id": "EDGAR_0000320193_10Q_20240202", "span_start": 295, "span_end": 433}
      ]
    }
  ]
}
```

## Checking a unit yourself

```bash
# Card schema, track, split and canary checks — standard library only.
python .github/validate_units.py analysis --stdlib-only

# The same, plus manifest checksums and the public-safety firewall (needs the shared toolkit).
python .github/validate_units.py analysis

# Faithfulness preview for one answer against one unit.
python faithfulness/judge.py --answer /tmp/answer.json --unit units/t4-EXAMPLE-eps-beat
```

`--unit` is the unit *directory*, not `corpus/`: the roster and target schema come from
`task.json`, the thresholds from `card.toml`, and a `doc_id` resolves only to a document the
manifest declares.

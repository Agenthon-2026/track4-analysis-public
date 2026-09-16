# Track 4 — Prediction Families (Public Reference)

## Executive summary (read this first)

Every Track 4 unit hands your agent a table of entities, a target to predict for each row, and a
frozen corpus of documents dated on or before the unit's `cutoff_date`. A **family** is the shape
of that pairing: what the rows are, what the target means, and which documents can settle it. This
page describes the families that actually ship in this repository, each one read off the unit's own
`card.toml` and `task.json` rather than restated from memory, so you can see the range of shapes
before you design a retrieval strategy. It is an **illustrative taxonomy, not a roster**: the
held-out set spans more families than are described here, and most have no published counterpart.
Nothing about an undocumented family changes what your container must do — the submission verb, the
answer schema, the gates and the composite are identical everywhere. If a term here is new, read
[`CONCEPTS.md`](CONCEPTS.md) first; for how a single unit is laid out on disk, read
[`AUTHORING-GUIDE.md`](AUTHORING-GUIDE.md).

---

## What every family has in common

- **Rows are entities.** Companies, issuers, Treasury maturities, futures markets, CPI components,
  macro series — one row per thing you predict for, keyed by `entity_id`.
- **One target per row**, declared as `target.type` in `task.json`: `classification`, `regression`
  or `ranking`. The same value is mirrored under `[scoring.params]` in `card.toml`, and your
  `answer.json` must repeat it in `target_type`.
- **A prediction interval on every row**, at the level the card pins (`interval_level`), regardless
  of target type.
- **Citations on every row**, into documents the unit's `manifest.json` declares, all dated on or
  before `cutoff_date`.
- **Evaluation uses the supplied task and frozen corpus.** Every published prompt says "Using ONLY
  the frozen evidence corpus". There is no open internet at scoring time, and post-cutoff
  citations are refused. The separate [training policy](TRAINING-POLICY.md) governs eligible
  offline training; it does not authorize additional evaluation inputs or citations.

## The families published here

Read directly from the shipped units (`units/*/card.toml`, `units/*/task.json`), 2026-08-28:

| Family slug | Target type | Target | Rows | What one row asks |
|---|---|---|---|---|
| `eps_beat_consensus` | classification | `eps_outcome` (`beat`/`miss`/`inline`) | 1 | Will this company's diluted EPS beat, miss, or land inline with consensus, at the card's threshold? |
| `eps_yoy_direction` | classification | `eps_yoy_direction` (`up`/`down`) | 6 | Will GAAP diluted EPS for the quarter reported after the cutoff be higher or lower than the same quarter a year earlier? |
| `eps_growth_regression` | regression | `eps_yoy_growth_pct` | 8 | By what percent will GAAP diluted EPS grow year over year for the quarter reported after the cutoff? |
| `credit_event` | classification | `credit_event_12m` (`credit_event`/`no_event`) | 8 | Will this issuer hit a bankruptcy filing, a payment default, or a rating-based credit event within twelve months of the cutoff? |
| `post_earnings_reaction` | classification | `earnings_reaction` (`positive_reaction`/`negative_reaction`/`flat`) | 3 | Which way does the market-adjusted one-day abnormal return go around the earnings release, against the card's flat threshold? |
| `rate_curve_cross_section` | regression | `yield_change_bps_intermeeting` | 6 | How many basis points does this constant-maturity Treasury yield move between the cutoff close and the resolution close? |
| `cpi_component_nowcast` | regression | `cpi_component_mom_first_print` | 11 | What is this CPI-U component's seasonally adjusted month-over-month percent change, as first printed? |
| `macro_revision_direction` | classification | `next_estimate_revision_direction` (`up`/`down`) | 12 | Will the agency's next published estimate of this reference month revise up or down? |
| `auction_demand` | regression | `bid_to_cover_ratio` | 7 | What bid-to-cover ratio does this announced Treasury coupon auction print? |
| `positioning_shift` | ranking | `net_positioning_change_pct_oi_rank` | 10 | Where does this futures market sit when the ten are ordered by change in net positioning as a share of open interest? |

Two units share `rate_curve_cross_section` (one around a 2022 FOMC meeting, one around a 2024
meeting), which is why eleven units cover ten family slugs.

**The `family` field is a descriptive slug, not an enum.** `taskcard.schema.json` constrains
`id`, `track`, `title` and `split`; it does not constrain `family`. Do not build a dispatch table
keyed on it and assume you have seen every key.

---

## Reading a family: the four things that decide your strategy

### 1. What settles the target, and when

Every unit carries a `cutoff_date` and a `resolution_date`. The gap between them is the horizon,
and it varies by more than an order of magnitude across the published set — from two days
(`t4-postearn-20240201-megacap`, 2024-01-31 to 2024-02-02) to twelve months
(`t4-credit-event-2023`, 2023-03-31 to 2024-03-31). Do not assume a minimum lag: read
`cutoff_date` and `resolution_date` off the unit and let the horizon shape how much weight you put
on slow-moving fundamentals versus the most recent filing.

The document that settles the target is, by construction, dated after the cutoff and therefore
absent from the corpus. On an EPS unit that is the quarterly report; on a credit unit it is the
8-K or docket entry announcing the event; on a rate unit it is the yield series at the resolution
close. You are being asked to forecast it, not to find it.

### 2. What the rows have in common — and what they do not

Some families give every row the same document set (a rates cross-section reads one FOMC statement
and one snapshot for all six maturities); others give each row its own subtree (`corpus_ref` points
at that entity's filings). Where rows share documents, evidence that supports one row usually does
not support another: the faithfulness gate builds one hypothesis per roster entity and only the
citations attached to *that* entity can support it. Pooling one strong citation across every row is
structurally unable to help.

### 3. Which columns are given and which are the answer

Feature columns are the pre-cutoff facts the unit hands you: `prior_year_q_eps`,
`start_yield_pct`, `net_pct_oi_20241022`, `latest_precutoff_estimate`, `offering_amount_usd_bn`.
Resolved outcomes are never among them. A column that looks like the answer is a *prior* value,
usually the same quantity one period earlier and can seed a persistence forecast. The regression
scorer instead compares error against the realized cross-entity mean, which is unavailable at
prediction time. The shared toolkit's `predictive_quality` function owns that calculation.

### 4. What units the target is in

Regression targets are in the prompt's stated units — basis points for
`yield_change_bps_intermeeting`, percent for `eps_yoy_growth_pct` and
`cpi_component_mom_first_print`, a bare ratio for `bid_to_cover_ratio`. Several units carry an
explicit `unit`/`units` column per row. Put that number, in those units, in `point_forecast`, and
put the interval on the same quantity. An interval on a probability never covers a change in basis
points.

---

## Ranking families: what is scored

On a ranking unit the ordering is taken from `point_forecast`, and `label` is not read. Put the
predicted **metric value** there — the thing the ordering is by. The optional integer `rank` field
records your ordering for a human reader and does not feed the score; if you supply it, it must be
a permutation of 1..n over the full roster. Putting the rank integer in `point_forecast` inverts
the ordering against a metric where larger is better; `README.md` gives the measured cost.

---

## Intervals: coverage is a target, not a quantity to maximise

The calibration leg of the composite is a **penalty on distance from the interval level**:

    composite = w_acc x predictive_quality - w_cal x |interval_coverage - interval_level|

Coverage is therefore *not* something to maximise. Coverage of 1.0 is penalised exactly as far from
the target as coverage of 0.80 is, and pushing empirical coverage past the level costs you score.
There is no width, sharpness or Winkler term anywhere in the scorer, so narrowness earns nothing
either — which means "submit a trivially wide interval" is not the mistake it is often assumed to
be, and neither is "submit a tight one". The single thing the scorer asks of your intervals is that
across the task set the truth falls inside them about as often as the level says. `README.md` and
`CONCEPTS.md` carry the measured single-unit numbers.

---

## Information hygiene: the two behaviours every family tests

These are properties of the whole track, not a separate family of units, and a unit does not
announce that it is testing them.

**Stale evidence.** A citation to a document dated after the unit's `cutoff_date` fails the unit,
whatever the prediction was. Apply a strict `doc_date <= cutoff_date` filter *inside* retrieval, not
as a post-hoc filter on the citations you emit — by then a stale passage has already shaped the
answer. The internal reason is `t4.citation_post_cutoff`, which maps to the `T4_STALE_EVIDENCE`
label (`t4.stale_evidence`).

**Corpus over priors.** Ground truth is what the corpus supports, and the judge is asked whether a
cited span entails the *prediction you submitted*, not the sentence you wrote about the span. An
agent that recalls what a well-known company usually does, then reaches for a passage to dress the
recollection up, contradicts its own citation — which is precisely what the entailment check
detects. Read the documents; predict what they support.

---

## The failure labels you can actually receive

From `qfbench2_track_analysis/codes.py` and `qfbench2_common.failure_labels`:

| Label | Value | Raised when |
|---|---|---|
| `T4_UNFAITHFUL_CITATION` | `t4.unfaithful_citation` | a citation is malformed or unresolvable, or too few roster predictions are entailed by their own cited spans |
| `T4_STALE_EVIDENCE` | `t4.stale_evidence` | a cited document is undated, or dated after `cutoff_date` |
| `T4_MISCALIBRATED_INTERVAL` | `t4.miscalibrated_interval` | an interval is structurally invalid — **not** a judgement that an interval was too wide or too narrow |
| `SCHEMA_INVALID_OUTPUT` | `shared.schema.invalid_output` | the answer fails the published schema, or the roster does not line up exactly (missing, unknown or duplicated `entity_id`) |

`T4_MISCALIBRATED_INTERVAL` is the one most often misread. It comes from
`T4Reason.INTERVAL_INVALID`; interval *width* is never a failure, it only moves
`interval_coverage`.

---

## Mistakes that actually cost score

1. **Citing a number of the wrong kind** — a revenue figure cited to support an EPS prediction, a
   headline index cited to support a component-level one. The premise does not entail the
   hypothesis, and the entailment check is asked about the hypothesis.
2. **Reading a granted waiver as an ongoing breach**, or an auditor's going-concern language as
   management's own statement. Credit units turn on this distinction.
3. **Predicting in the wrong units** — a probability where the prompt asks for basis points, a
   ratio where it asks for percent. The interval then cannot cover the truth either.
4. **Putting the rank integer in `point_forecast`** on a ranking unit.
5. **Emitting fewer rows than the roster**, or an extra one. The roster must line up exactly; a
   mismatch is a whole-submission schema failure, not a partial-credit deduction.
6. **Leaving out `interval.lo`/`interval.hi` or `claims` on a single row.** Same consequence: the
   whole submission fails `g1_schema`.
7. **Filtering stale documents only at citation time.** See above.

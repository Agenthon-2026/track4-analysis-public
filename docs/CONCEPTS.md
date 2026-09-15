# Track 4 — Concepts Explained in Plain English

## Executive summary (read this first)

This file defines every concept you need to understand Track 4 (Explainability). Track 4
presents an AI with a **table of entities** — companies, securities,
or events — and asks it to predict a target per row, grounded in a frozen evidence corpus of
real documents. If the AI cannot prove its predictions came from those documents, its submission
is rejected. This file explains, from scratch, what tabular data is, what the difference between
classification, regression, and ranking is, what tabular foundation models are, how the NLI
model checks citations, what calibration means for a 90% confidence interval, and what the
adversarial traps are designed to catch. This file is the definitive glossary for Track 4; the
competition-wide GLOSSARY spanning all four tracks is published with the shared toolkit, at
`Agenthon-2026/Agenthon2026-public`, file `docs/GLOSSARY.md` — the same public repository this
track installs `qfbench2-common` from. For the families published here see `CATEGORIES.md`, and
for the on-disk layout of a unit see `AUTHORING-GUIDE.md`.

---

## Tabular data vs. time-series data — the key distinction

**Time-series data** is a sequence of observations of the same thing over time — for example,
the daily closing price of Apple stock, or quarterly GDP. The challenge is to predict the next
value in that sequence.

**General tabular data** is structured as a table where each **row** is an entity (a company,
a security, an event) and each **column** is a feature (a property of that entity). The columns
can be a mix of:
- **Numeric**: market cap, revenue growth, debt ratio, EPS consensus
- **Categorical**: sector, credit rating, country
- **Text**: management commentary excerpt, filing summary

The challenge in tabular prediction is to predict a **target value** for each row, using all the
columns as features.

Track 4 uses general tabular data. Track 2 uses time-series data. Both add text evidence and
agentic reasoning, run closed-resource (Docker, restricted network: no open internet — the only
permitted egress is through the organizer's audited proxy to the organizer-hosted model endpoint,
and vendor model APIs are refused by the proxy), and enforce leakage controls. `SUBMISSION_CLI.md`
is the authoritative statement of the network contract.

---

## Rows, columns, and the target

In Track 4, every task gives you:

- **Rows** = entities to predict for (e.g., three companies: Apple, Microsoft, Alphabet).
- **Columns** = features describing each entity (e.g., sector, consensus EPS, revenue growth).
- **Target** = what you must predict per row (e.g., whether each company will beat its EPS
  estimate, or rank the companies by their predicted sector return).

A tiny example table for an EPS-beat task:

| entity_id | sector | consensus_eps | rev_growth_3q | target |
|-----------|--------|---------------|---------------|--------|
| AAPL | Info Tech | 1.50 | 0.08 | ??? |
| MSFT | Info Tech | 2.82 | 0.15 | ??? |
| GOOGL | Comm. Svcs | 1.64 | 0.11 | ??? |

The agent fills in the `target` column for each row by reading the frozen corpus for each entity.

---

## Classification, regression, and ranking

**Classification**: the target is one of a fixed set of class labels. Example: `beat`, `miss`,
or `inline` for an EPS task. You output one label per row. Success is measured by accuracy
(fraction of rows where the predicted label matches the true label).

**Regression**: the target is a number. Example: the probability that a company experiences a
credit event (a number in [0, 1]), or the expected yield-curve change in basis points. You output
a number per row. Success is measured by error metrics like MAE (mean absolute error), or a skill
score that compares your error to a naive baseline's error.

**Ranking**: the target is an ordering across the rows. Example: rank three sectors from highest
to lowest predicted relative return. You output a rank position per row. Success is measured by
rank correlation (how well your predicted ordering matches the true ordering).

Track 4 tasks declare their target type as `target.type` in `task.json`: `classification`,
`regression`, or `ranking`. It is *not* a flat `target_type` key -- that name belongs to the
`answer.json` you emit, and to `[scoring.params]` in `card.toml`. The scorer handles each type
differently (see `scoring/scoring.py`).

---

## Evidence corpus

The **evidence corpus** is the complete, frozen set of documents available to the agent for one
task. Every document has a `doc_date` (the date it was filed or published) and a `doc_id`
(a unique identifier). The corpus is assembled before the task is issued and does not change.

"Frozen" means: the corpus is locked in place at task-generation time. No documents can be added
or removed once the task enters the test set. This ensures every agent sees exactly the same
evidence.

---

## SEC filings and EDGAR

The **SEC** (US Securities and Exchange Commission) requires all public companies to file regular
reports. The SEC's public database of filings is called **EDGAR**. Track 4 uses three filing
types:

**10-K (Annual Report)**: Filed once per fiscal year. The most comprehensive document: full
audited financial statements, management discussion, risk factors, and an outlook for the coming
year.

**10-Q (Quarterly Report)**: Filed after each of the first three fiscal quarters. Shorter than
the 10-K, unaudited. Contains three months of financials and management commentary. This is the
main source of EPS guidance and segment revenue data.

**8-K (Current Report)**: Filed within four business days of any "material event" — announcing
preliminary earnings, entering a major contract, changing the CEO, disclosing a legal settlement,
or reporting that the auditors have serious doubts about the company's ability to continue. 8-Ks
vary hugely in importance; some are routine, some are alarming.

**FRED (Federal Reserve Economic Data)**: A public database of economic time series maintained
by the Federal Reserve Bank of St. Louis, covering interest rates, inflation (CPI, PCE),
employment, and many other indicators. In Track 4, FRED data is included as snapshot documents
in the corpus.

---

## Tabular foundation models — the text-blind baselines

A **tabular foundation model** is a machine-learning model designed specifically for tabular data
(rows and columns of mixed numeric/categorical features). Unlike a language model, it does not
read text — it operates only on the numeric and categorical feature columns.

**TabPFN** (Tabular Prior-Fitted Networks): a transformer model trained on thousands of synthetic
tabular datasets. It can make good predictions on small tables (a few hundred rows) without any
task-specific training — you just give it the feature table and it outputs predictions. It is the
state-of-the-art text-blind baseline for small cross-sections.

**Gradient boosting (XGBoost / LightGBM)**: a classical machine learning method that builds an
ensemble of decision trees, one at a time, each correcting the errors of the previous one. It is
the dominant method for tabular data in industry and competitions. Very fast and often competitive
with deep learning on structured data.

Both are **text-blind**: they cannot read the evidence corpus. They are included in Track 4
baselines to show the floor. If your agent's predictive quality does not exceed TabPFN or
gradient boosting, the text evidence and agentic reasoning have added no value.

---

## Citation and evidence trace

A **citation** in Track 4 is a precise reference to a specific passage in a specific document:

```json
{
  "doc_id": "EDGAR_0000320193_10Q_20240202",
  "span_start": 295,
  "span_end": 627,
  "claim": "Apple's Services revenue grew 11% year-over-year in Q1 FY2024."
}
```

- `doc_id`: which document (format: `{source}_{identifier}_{date}`).
- `span_start` and `span_end`: character offsets (position numbers in the document text) that
  define exactly which passage is cited. Zero-indexed.
- `claim`: the sentence the agent is asserting, in its own words.

The scoring pipeline resolves the character offsets to extract the actual passage text, then
checks whether that passage entails the claim.

The **evidence trace** is the collection of all citations in a submission: the chain of evidence
from document passages to each prediction made by the agent.

---

## Faithfulness

**Faithfulness** is the property that every *prediction* in an answer is supported by the evidence
cited for it. It is not a property of the sentences you write: the hypothesis put to the judge is
built from the values you submitted — `label` / `point_forecast` / `rank` / `interval` — plus the
trusted task schema (`qfbench2_track_analysis/hypothesis.py`). Your `claim` text is parsed and
reported, but it is never what the judge is asked about, so describing a cited passage accurately
cannot make a wrong prediction faithful.

Faithfulness is checked automatically by an NLI model. A roster entity's prediction is
**supported** when the ensemble's two-way entailment score for (cited passage, that prediction) exceeds the
per-citation threshold of 0.5 (`tau_citation`) for at least one span cited *for that entity*.

**The denominator is the entity roster, not the number of claims you wrote.** One hypothesis is
built per roster entity, in trusted roster order, so padding a submission with extra prose cannot
move the score, and a citation can only support the entity it was attached to. The faithfulness
score is the fraction of roster entities whose prediction is supported. A submission must score at
or above `faithfulness_threshold` = 0.80 — at least 80% of the roster's predictions supported — to
be eligible for ranking. Submissions below that are ineligible regardless of whether their
predictions were correct.

---

## NLI / entailment — with a worked example

**NLI** stands for **Natural Language Inference** (also called **textual entailment**). Given a
**premise** (a passage) and a **hypothesis** (a claim), does the premise **entail** the hypothesis
(i.e., if the premise is true, must the hypothesis also be true)?

The models distinguish three NLI classes:

- **Entailment**: the premise clearly supports the hypothesis.
- **Neutral**: the premise neither confirms nor denies the hypothesis.
- **Contradiction**: the premise says the opposite of the hypothesis.

Track 4's retained judge score compares only the entailment and contradiction logits
(the model's raw class scores): `exp(entailment) / (exp(entailment) + exp(contradiction))`.
Neutral is excluded from this normalization. A high score therefore favors entailment over
contradiction but does not establish a low neutral probability; neutral examples have no
fixed score range under this calculation.

The cited passage is the premise. The prediction checker builds the hypothesis from the
submitted prediction and trusted task schema, as described under "Faithfulness" above.
The examples below illustrate the three NLI meanings, rather than measured model scores.

**Tiny example:**

> Premise: "Services net sales were $23,117 million for the first quarter of fiscal 2024,
> compared to $20,766 million for the same period in fiscal 2023."

> Hypothesis 1: "Services revenue rose year-over-year in Q1 FY2024."
> Verdict: **Entailment** — $23,117M > $20,766M.

> Hypothesis 2: "Services revenue grew faster than iPhone revenue."
> Verdict: **Neutral** — the passage says nothing about iPhone revenue.

> Hypothesis 3: "Services revenue declined in Q1 FY2024."
> Verdict: **Contradiction** — the passage shows growth.

---

## DeBERTa judge ensemble

Track 4 uses an **ensemble** of two DeBERTa (a type of transformer language model) NLI models:
- `cross-encoder/nli-deberta-v3-large`
- `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`

Both models are trained on large NLI datasets (MNLI, FEVER-NLI, ANLI), following Laurer et al.
(2024). The ensemble averages their two-way entailment scores. Averaging does not turn these
values into three-way entailment probabilities or establish calibration. Both models run offline (no internet access)
from pre-cached weights inside the evaluation Docker image.

The judge passes one hypothesis as the sole candidate label, with `hypothesis_template="{}"`
and `multi_label=True`. In the pinned Transformers pipeline, `multi_label=False` with one
candidate takes the same entailment-versus-contradiction branch; it does not produce a
constant 1.0. See the [Transformers implementation](https://github.com/huggingface/transformers/blob/5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9/src/transformers/pipelines/zero_shot_classification.py#L235-L254)
and the call in [`faithfulness/judge.py`](../faithfulness/judge.py).

The threshold for a single citation to pass is 0.5 (`tau_citation`): the entailment
score, averaged across both models, must exceed 0.5. Admission then requires at least
80% of the **roster's** predictions to be supported (`faithfulness_threshold` = 0.80) — see
"Faithfulness" above for why the denominator is the roster and not the claim count.

---

## Embargo and cutoff

The **cutoff date** (`cutoff_date`) is the latest date on which a document may be dated to be
included in the corpus. Every document has a `doc_date`; the rule is `doc_date <= cutoff_date`.

The **embargo** is the mechanism that enforces this rule. Documents dated after the cutoff are
blocked. The evaluation harness checks every `doc_id` the agent cites against the corpus manifest
and flags any citation whose `doc_date > cutoff_date`. This is called a `T4_STALE_EVIDENCE`
violation.

Why does this matter? Because a post-cutoff document often contains the answer. On an EPS task
the post-cutoff 10-Q literally reports the EPS being predicted; on a credit-event task a
post-cutoff 8-K may announce the very default being forecast. Citing those documents is
reading the answer key.

---

## Predictive quality — classification, regression, ranking

The way we measure prediction accuracy depends on the task type:

All three land in **[0, 1]**. That is not incidental: the composite's first leg is
`w_acc x predictive_quality`, and the domain of the whole metric depends on quality never going
negative.

**Classification accuracy**: the fraction of rows where the predicted label (`beat`, `miss`, etc.)
matches the true label. A random guesser on a three-class problem achieves ~0.33.

**Regression skill score**: `clamp(1 - MAE / baseline_MAE, 0, 1)`, where `baseline_MAE` is the
error of predicting the **cross-entity mean of the realized values** for that unit. A perfect model
scores 1.0; a model no better than that mean scores 0.0; and a worse-than-baseline model scores
0.0 as well — the score is clamped at zero and does **not** go negative.

**Ranking (Spearman correlation, rescaled)**: the Spearman rank correlation `rho` between your
ordering and the true one, rescaled as `(rho + 1) / 2`. **Ties rank as ties** — tied values share
the mean of the positions they occupy — so an answer that expresses no ordering cannot inherit the
roster's own order and be graded on it as if it were a prediction. A perfect ranking scores 1.0; a
random one about 0.5; a perfectly reversed one 0.0. A unit with fewer than two rows scores 0.5, and
so does a **constant** `point_forecast`: with no rank variance `rho` is 0, which rescales to the
neutral 0.5 — neither rewarded nor punished for saying nothing. An answer that predicts nothing at
all scores 0.0.

**Missing rows are not dropped.** A prediction you omit, or emit as NaN, is scored worst-case for
that row — wrong for classification, at the baseline for regression, ranked last for ranking — so
answering only the rows you are confident about can never raise your quality.

---

## 90% interval, coverage, and calibration

Every Track 4 submission must include a **90% confidence interval** [lo, hi] around the point
forecast. The interval is supposed to mean: "I am 90% confident the true value will fall between
lo and hi."

**Coverage** is the fraction of rows (or tasks) for which the true value actually fell inside
[lo, hi]. A well-calibrated agent should achieve empirical coverage close to 90%.

**Calibration** is the alignment between stated and empirical confidence. A perfectly calibrated
agent's 90% intervals contain the true value exactly 90% of the time.

**Sharpness** — narrowness — is a standard companion to calibration, and **Track 4 does not score
it.** There is no width term, no Winkler score and no sharpness term anywhere in the scorer or the
shared toolkit; a sweep for `sharpness|winkler|brier|interval_width` over every `.py` file returns
nothing, while the control term `interval_coverage` is found where expected.

Earlier versions of this page described a normalised Winkler score, with a 20x miss penalty and a
baseline calibrated to 0.5. **None of that was ever implemented.** The whole calibration leg is:

    composite = w_acc x predictive_quality - w_cal x |interval_coverage - interval_level|

so the only thing an interval's width does is move empirical coverage toward or away from 0.90.
Measured on a three-entity unit, everything identical except the interval: `[1.4, 1.9]` scored
**-0.037** at coverage 0.0, and `[-1e9, 1e9]` scored **+0.203** at coverage 1.0. On a single unit,
widening strictly helps. Aim for coverage near 90% across the task SET; do not expect to be paid
for narrowness.

---

## Stale-evidence trap

The **stale-evidence trap** is an adversarial variant in which documents dated after
`cutoff_date` are present in the corpus. The trap tests whether the agent notices the date and
declines to use them. Assume nothing about how many such documents a unit holds, where they
rank in retrieval, or what they say: the only safe rule is the one the gate enforces, which is
that a citation to any document dated after the cutoff fails the unit.

An agent that cites a stale document fails the unit outright — even if its directional call was
accidentally correct. There is no `embargo_ok` flag in any answer or outcome file (a sweep for it
over every `.py` file returns nothing); the scorer raises `t4.citation_post_cutoff`, which maps to
the `T4_STALE_EVIDENCE` failure label and the domain minimum. The correct behaviour is to detect that the document's
date exceeds the cutoff and ignore or explicitly flag it.

---

## Counterfactual variant

The **counterfactual variant** alters the corpus itself — a value in a document is changed so
that the evidence points somewhere other than the real-world outcome. The agent is not told
what was changed, or whether anything was. Ground truth is what a corpus-faithful agent should
conclude from the documents it was given, not what happened in the world.

This tests whether the agent is genuinely reading the documents or using the corpus as a post-hoc
citation exercise to justify a conclusion drawn from prior knowledge. An agent anchored on prior
beliefs will contradict the corpus text in its own claims — a contradiction the NLI judge detects.

---

## Manual review

The automated NLI check is the primary faithfulness gate. But it is not perfect. **Manual review**
is triggered in two situations: (1) any submission that scores in the top 20% of the leaderboard
gets human review to confirm the automated scoring did not miss a subtle faithfulness failure;
(2) any submission where the NLI score for a key citation falls between 0.40 and 0.60 — within
0.10 of the 0.5 per-citation threshold (the "borderline zone") — gets a human reader.

Two finance-domain reviewers work independently. If they agree, their verdict stands. If they
disagree, the track lead makes the final call. Reviewers can override an NLI score only when the
NLI model is clearly wrong because of highly domain-specific financial terminology — and they must
document the specific terminology issue.

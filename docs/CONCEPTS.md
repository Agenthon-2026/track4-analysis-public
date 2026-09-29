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
checks whether that passage entails the claim. Offsets must name a real slice of the document
(`0 <= span_start < span_end <=` the document's length); a citation whose offsets do not is not
clamped, it names no passage and supports nothing (scorer 5.1.1). The judge reads at most its
tokenizer window, 512 tokens for the passage and the claim together, and the passage is cut to
that window before the numeric backstop and the judge read it: a figure past the window does not
count. Cite the passage that carries your figures, not a whole document.

The **evidence trace** is the collection of all citations in a submission: the chain of evidence
from document passages to each prediction made by the agent.

---

## Faithfulness

**Faithfulness** is the property that no claim in an answer is false or misattributed. From scorer
5.2.0 it is a **per-claim penalty**, not an admission gate: nothing has to be "passed", nothing is
earned, and a demonstrably false claim costs its share of the unit's score. Whether your evidence
supports your *forecast* is not this check's question; that belongs to reasoning grading, which is
the only place evidence earns credit.

**The rule.** Each claim is either **false** or **neutral**. The unit's score is

    score = composite x (1 - F / (F + min(T, 3 x E)))

where F is the number of false claims, T the number of other claims and E the number of entities
in the unit (the "soft floor"; `penalty_k` = 1 is the power the factor is raised to). In plain
terms: each false claim costs a share of the unit; other claims beyond 3 × E in total (three times
the number of entities, counted over the whole unit, not per entity) do not dilute the
cost; a unit with no false claims is not penalised; content-free claims neither earn nor cost.
Up to that cap it is the plain share (on a unit with 7 or more entities, one false claim among
twenty costs 5%; on a 1-entity unit the cap is 3, so the same claim costs 1/(1 + 3) = 25%);
past that, a false claim keeps costing at least 1/(F + 3E) of the unit however many claims are
added. A unit whose every claim is false scores 0. A unit with no claims
keeps its whole composite (alignment still requires `claims[]` on every entity). A unit is refused
(worst-case score) only for structural errors: a schema violation, a missing, extra or duplicated
entity, a non-finite number, an unresolved, undated or post-cutoff citation, a malformed citation.

A claim is **false** when any one of these holds, checked in this order:

1. **Wrong entity.** Every corpus document carries, in the unit's trusted manifest, either the
   roster entities it is about (`entity_ids`) or `shared: true` for a market-wide document (an
   FOMC statement, a macro series). A claim for entity E that cites a document the manifest does
   not list E on, and does not mark shared, is false. A document about someone off the roster
   (`entity_ids: []`) is about nobody on it. Before 5.2.0 one such citation refused the whole
   unit; now it costs that claim. You can verify this rule yourself from the manifest.
2. **Out-of-range citation.** A citation whose `span_start` / `span_end` are not a real slice of
   the document (negative start, end past the document, start at or after end) names no passage,
   and the claim carrying it is false.
3. **Malformed claim.** An empty claim, one over 4000 characters, or one longer than **400
   judge tokens** (counted with the judge's own tokenizer) has no usable text. A longer claim would leave the judge too little of its 512-token window for
   the passage it cites.
4. **A figure its passage does not carry.** Exact code, no model: **every figure in a claim must
   appear in a passage the claim cites**, read against the whole cited span
   `text[span_start:span_end]`, not only the part the judge reads. A claim with a figure none of
   its cited spans carries is false. Details:
   - Dates, years, periods, ordinals, identifiers, form or item numbers, and counts of periods
     ("13 weeks", "10 sessions", "2 years") are not figures.
   - A number that is part of one of the **unit's own entity names or tickers** as `task.json`
     writes them ("Phillips 66", "S&P 500", "3M") is not a figure. Any other name is read as
     written: a number in it is a figure.
   - A figure that **exactly** equals a scored value you submitted is exempt: your point
     forecast on a regression or ranking unit, and your interval bounds only when the unit's
     interval leg is scored. Your rank is never exempt, and no scale, percent-versus-ratio or
     rounding tolerance applies to your own values. Nothing else is exempt; a value from the
     task table must cite the task table (see "Citing the task table" below).
   - For figures in a passage, separators, scale (`$8.5M`, `1,498,614` in a table headed "in
     thousands"), percent-versus-ratio, sign and rounding to the precision you wrote are all
     tolerated.
   - A cited span longer than **8,000 characters** (the per-citation cap of the reasoning
     contract) anchors no figure. Cite the passage that states your figures, not a whole filing.
   - A claim that is **word for word** a piece of a span it cites (whitespace runs compared as
     one space) passes the figure check whole, even if the quote is cut mid-number. This wins
     over the span cap above: a verbatim quote passes even when the span it cites is over 8,000
     characters (the quote is looked for in the first 200,000 characters of the span); the cap
     applies to every other claim.
5. **Contradicted by its passage.** For every other claim the NLI judge reads each cited passage
   (the premise) against **your `claim` text** (the hypothesis) and returns the three-way
   probability that the passage **contradicts** the claim, averaged over the ensemble's two
   models. The claim is false when that probability exceeds `contradiction_bar` = 0.9 for any of
   its cited passages. A verbatim quote of a passage it cites (as in 4) states what that passage
   states and is not put to the judge.

Every other claim is **neutral**: a quoted passage, an accurate paraphrase, and generic text the
passage neither confirms nor denies are never charged and earn nothing here. Beyond 3 × E claims
in total (E = the number of entities in the unit), extra claims do not dilute the cost of a false
claim either.

### What the check measures, and what it does not

It establishes that each claim is about the right entity, that it cites a real passage, that its
figures are in that passage, and that the passage does not say the opposite. It does not
establish that the prediction was derived from the evidence, and it does not try to: content-free
claims are neutral here and score near zero in reasoning grading, so saying nothing checkable
earns nothing, and stating specific true facts costs nothing.

**Why the judge is asked about your claim and not about your forecast.** The obvious alternative
is to ask whether the cited passage entails the *prediction*, rendered as a sentence from your
submitted values. That question has no right answer: every document in a unit's corpus predates
the unit's cutoff, and your forecast is about what happens after it, so no passage can entail one.
It is still computed and recorded for the organizer's review queue as `prediction_relevance`, and
it never affects your score.

**Why contradiction, and why a high bar.** Before 5.2.0 a claim had to be *entailed* (two-way
score above 0.5) and 80% of claims had to pass, so a vague claim could pass while one weak but
honest claim could refuse the unit. Generic text is neither entailed nor contradicted, so asking
for contradiction leaves it neutral. The bar is 0.9 because lower bars flag long verbatim quotes
that the judge's 512-token window cuts short.

**Why the entity check is deterministic and comes first.** An NLI model has no notion of which
company a passage is about: give it a claim about company A and a passage from company B's filing
and it answers "entailed" whenever the wording matches. The cited *document* does carry that
identity, in the manifest, so the correspondence is checked there by exact code. On units whose
corpus is shared by design (macro and rate units), every document is `shared`, so only the other
checks apply.

**Claims are extractive facts; computed figures belong in `submitted_reasons`.** A change
between two rows, a growth rate, an average, a ratio, a spread, a share, a sum: any number you
calculated is your argument, not the evidence. State the inputs as claims, each with the figures
its passage carries, and put the calculation in a reason's `mechanism`, where reasoning grading
judges derivations. A `submitted_reasons` block that does not match the schema (an empty list,
more than 3 reasons, or a reason missing a required field) makes the whole answer invalid, like
any schema error, so run the local checker (`check_submitted_reasons`) first; leaving reasons out
never costs anything.

- Wrong (claim): "Deposits grew 3.1% quarter on quarter, from $41.2 billion to $42.5 billion."
  (3.1% is computed; no passage states it.)
- Right: claim "Total deposits were $41.2 billion at 2025-12-31." (cites the row that says so);
  claim "Total deposits were $42.5 billion at 2026-03-31." (cites its row); reason mechanism
  "Deposits rose about 3.1% quarter on quarter (42.5 / 41.2 - 1) ...".

**Why "every figure".** Under the per-claim penalty a false claim costs a share of the unit
(at least 1/(F + 3E) of it: one false claim among twenty costs 5% on a unit with 7 or more
entities), not the whole unit, so the check can ask for what an
honest extractive claim always satisfies: every amount it states is one its evidence states. The
earlier rule (false only when *no* figure was present, read in the judge's window) let a
fabricated figure pass beside a genuine one.

### Citing the task table

A figure the task gives you (a value in a row of `task.json` `entities`, such as a prior-period
balance) is cited like a corpus passage, with the reserved document id `"task"`:

```json
{"doc_id": "task", "span_start": 480, "span_end": 956}
```

The text a `"task"` span indexes is the **task table**: one line per `entities` row, in the order
of `task.json`, each row written as

```python
json.dumps(row, ensure_ascii=False, separators=(", ", ": "))
```

(keys in file order), the lines joined by a single `"\n"` with no trailing newline. Offsets are
character (code point) offsets into that text, the same convention as a corpus document's `text`.
`qfbench2_track_analysis.corpus.task_table_text(task)` returns exactly this text and each row's
`[start, end)`. A task span must lie inside **one** row, and that row must be the row of the
entity whose prediction carries the claim: citing another entity's row, or a span that crosses
rows, is a wrong-entity citation (false). The task table is dated at the cutoff, so it never trips
the embargo. Only the `entities` rows are citable; the prompt and notes are instructions, not
evidence.

Every rule is specified case by case, on synthetic inputs, by the scorer's own test suite:
`scoring/tests/test_claim_penalty.py` (the penalty, its parameters, the structural refusals),
`scoring/tests/test_numeric_backstop.py` (every figure form),
`scoring/tests/test_every_figure_rule.py` and `scoring/tests/test_figure_rule_refinements.py`
(every figure, the task table, names, the span and claim-length caps, verbatim quotes) and
`scoring/tests/test_entity_bound_citations.py` (the entity rule).

**Why a cap of 3 × E claims.** With a plain share (false claims / all claims), padding an
answer with many content-free claims would shrink what each false claim costs toward nothing.
Counting at most 3 × E non-false claims in total (E entities) stops that: a false claim always costs at least
1/(F + 3E) of the unit, while an honest answer with no false claim still scores its whole
composite, and answers with at most 3 × E other claims score exactly as under the plain share.

**The parameters are fixed scorer constants.** `penalty_k` = 1, `contradiction_bar` = 0.9 and the
cap of 3 × E other claims
are the same for every unit and every entrant; a `card.toml` or plan that names either is
refused. Cards still carry the retired
`faithfulness_threshold` = 0.80, which 5.2.0 reads as "use the per-claim penalty" and nothing
else; any other value is refused. `tau_citation` = 0.5 now only sets the recorded
`prediction_relevance` diagnostic. These are fixed scorer constants that will not change without a
published notice.

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
and `multi_label=True`. In the linked Transformers 5.15.0 implementation, `multi_label=False` with one
candidate takes the same entailment-versus-contradiction branch; it does not produce a
constant 1.0. See the [Transformers implementation](https://github.com/huggingface/transformers/blob/5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9/src/transformers/pipelines/zero_shot_classification.py#L235-L254)
and the call in [`faithfulness/judge.py`](../faithfulness/judge.py).

From scorer 5.2.0 the faithfulness check asks a separate question of the same forward pass: each
model's **three-way** softmax over entailment, neutral and contradiction, of which the
contradiction probability is averaged across both models. A claim is false when that average
exceeds `contradiction_bar` = 0.9 for a span it cites — see "Faithfulness" above. The two-way
entailment score above is unchanged and is used only for the recorded `prediction_relevance`
diagnostic (bar 0.5, `tau_citation`).

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

**Classification and ranking are anchored to the naive rule (scorer 5.2.0).** The raw quality
below is mapped so that 0 stays 0, matching the unit's declared naive rule
(`reference/naive_answer.json`) scores **0.5**, and a perfect answer scores 1, linearly in
between on each side. The anchor is the stronger of the declared naive rule's quality and, on a
ranking unit, a constant forecast's 0.5. Regression already measures skill against a baseline
(below), and the interval leg is measured against the naive interval, so both are unchanged. The
raw quality, the naive rule's quality and the anchor are recorded in the unit's diagnostics.

**Classification accuracy**: the fraction of rows where the predicted label (`beat`, `miss`, etc.)
matches the true label. A random guesser on a three-class problem achieves ~0.33.

**Regression skill score**: `naive_MAE / (naive_MAE + MAE)`, where `naive_MAE` is the mean
absolute error of the unit's **declared naive rule** (`reference/naive_answer.json`) and `MAE` is
yours, both over the roster. An exact answer scores 1.0; an answer with the naive rule's error
scores 0.5; a worse answer falls toward 0 as its error grows, and the score does **not** go
negative.

**Ranking (Spearman correlation, rescaled)**: the Spearman rank correlation `rho` between your
ordering and the true one, rescaled as `(rho + 1) / 2`. **Ties rank as ties** — tied values share
the mean of the positions they occupy — so an answer that expresses no ordering cannot inherit the
roster's own order and be graded on it as if it were a prediction. A perfect ranking scores 1.0; a
random one about 0.5; a perfectly reversed one 0.0. A unit with fewer than two rows scores 0.5, and
so does a **constant** `point_forecast`: with no rank variance `rho` is 0, which rescales to the
neutral 0.5 — neither rewarded nor punished for saying nothing. An answer that predicts nothing at
all scores 0.0.

**You cannot omit a row at all.** The roster is the denominator, and alignment checks it before
any metric runs: your entity set must equal the trusted roster exactly — complete, unique, nothing
extra. A missing, duplicated or unknown `entity_id` is a participant failure
(`incomplete_output`), and the unit takes the worst value W rather than being scored on the rows
you did answer. Measured on a four-entity ranking unit: the full roster correctly ordered, with
intervals that cover, scores +0.8737; the same answer with one row omitted scores W = 0.0
(scorer 5.1.0; 5.0.0 measured +0.67 and −0.27). On the leaderboard W shows as −0.27
(leaderboard = −0.27 + 1.27 × analysis).

So "answer only the rows you are confident about" is not a strategy that scores badly — it is not
a submission. A NaN is refused on the same grounds rather than being scored worst-case: rows are
never dropped to make a denominator smaller.

---

## 90% interval, coverage, and calibration

Every Track 4 submission must include a **90% confidence interval** [lo, hi] around the point
forecast. The interval is supposed to mean: "I am 90% confident the true value will fall between
lo and hi."

**Coverage** is the fraction of rows (or tasks) for which the true value actually fell inside
[lo, hi]. A well-calibrated agent should achieve empirical coverage close to 90%.

**Calibration** is the alignment between stated and empirical confidence. A perfectly calibrated
agent's 90% intervals contain the true value exactly 90% of the time.

**The interval leg (scorer 5.1.0).** Each row's interval is scored with the interval score (the
Gneiting-Raftery interval score): its width plus `2/alpha` times the
distance by which the truth falls outside it, `alpha = 1 - interval_level` (so 20x the miss at
90%). The unit value is the mean over the roster, and it is compared with the same quantity for the
unit's declared naive interval (`reference/naive_answer.json`):

    interval_quality = naive_IS / (naive_IS + IS)
    composite = w_acc x predictive_quality + w_cal x interval_quality

`interval_quality` is 0.5 when your intervals score the same as the naive rule's, approaches 1 for
a sharp interval that contains the truth, and approaches 0 for a very wide one or a far miss.
Both legs lie in [0, 1], so the domain is `[0, 1]` and the worst value W is 0.0 (shown as -0.27
on the leaderboard, where leaderboard = -0.27 + 1.27 × analysis). Coverage is still
reported as a diagnostic, but no longer scored.

Measured on a three-entity unit, everything identical except the interval (naive band
`[0.5, 3.5]`): the naive band scored **+0.383**, `[1.4, 1.9]` (missing all three values) **+0.297**
and `[-1e9, 1e9]` **+0.233**. Under 5.0.0 the calibration leg was
`- w_cal x |interval_coverage - interval_level|`, the same two intervals scored **-0.037** and
**+0.203**, and widening strictly helped; that is no longer true.

A classification unit whose numeric truth is only a 0/1 label code may declare
`interval_leg = false` in its card's `[scoring.params]`; it is then scored on the label alone, as
a pure-label unit is. From scorer 5.2.0 such a unit's composite is the (anchored) prediction leg
itself, not `w_acc` times it, so it is not capped at `w_acc`. The interval is still required by the
answer schema on every unit.

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

The automated NLI check is the primary faithfulness check. But it is not perfect. **Manual review**
is triggered in two situations: (1) any submission that scores in the top 20% of the leaderboard
gets human review to confirm the automated scoring did not miss a subtle faithfulness failure;
(2) any submission where the NLI score for a key citation falls between 0.40 and 0.60 — within
0.10 of the 0.5 per-citation threshold (the "borderline zone") — gets a human reader.

Two finance-domain reviewers work independently. If they agree, their verdict stands. If they
disagree, the track lead makes the final call. Reviewers can override an NLI score only when the
NLI model is clearly wrong because of highly domain-specific financial terminology — and they must
document the specific terminology issue.

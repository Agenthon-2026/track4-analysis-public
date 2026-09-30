# Track 4 — Explainability (Public Starter Kit)

## Executive summary (read this first)

Track 4 asks: can an AI predict a label or value for each row in a table — for example, whether a
company will beat its earnings forecast — **and prove it used real documents to do so?** Every
submission receives a table of entities (companies, securities, events), each with a mix of numeric
and categorical features. The AI must predict a **target** per row (a class label for
classification tasks, a numeric estimate for regression, or a ranked order across rows for
ranking tasks), return a 90% confidence interval, and supply **citations** to passages in a frozen
evidence corpus that ground each claim. If the citations do not hold up under an NLI (natural
language inference) check — the automated system decides whether the cited passage actually
supports the claim — each claim the check finds false costs its share of the unit's score, however
accurate the predictions were.

This folder is the **public starter kit** — everything you need to build, test, and submit a
Track 4 agent. The hidden test questions and their ground-truth outcomes live in a private
repository that only organisers can see. **No answers are in this folder.**

New to the track? Read `docs/CONCEPTS.md` first — it defines every term used in Track 4 — then
this file, then work the exemplar unit and the ten practice units under `units/`. For the shape of
a unit on disk and which of its fields the scorer actually reads, see `docs/AUTHORING-GUIDE.md`;
for the prediction families published here, `docs/CATEGORIES.md`. (The competition-wide GLOSSARY
spanning all four tracks is published with the shared toolkit: `Agenthon-2026/Agenthon2026-public`,
file `docs/GLOSSARY.md` — the same public repository this track installs `qfbench2-common` from.
`docs/CONCEPTS.md` stays the definitive reference for this track.)

---

## What Track 4 is — and what it is not

Track 4 is about **general tabular prediction** grounded in text evidence. Each task gives you:

- A **table** where every row is an entity (a company, a security, an event) and every column is
  a feature — some numeric (revenue, market cap, spread), some categorical (sector, rating), some
  text (management commentary excerpts).
- A **target column** to predict per row: a class label (e.g., `beat`/`miss`/`inline`), a numeric
  value (e.g., a probability score), or a ranking across the rows.
- A **frozen evidence corpus** of SEC filings (10-K, 10-Q, 8-K) and macroeconomic releases (FRED
  data), all dated on or before a `cutoff_date`.

Track 4 is **not** time-series forecasting. It does not ask you to predict the next value in a
sequential series. It asks you to predict a target for a cross-section of entities, using tabular
features plus evidence-grounded reasoning.

| Track 2 | Track 4 |
|---------|---------|
| Data: time series (observations over time for one or a few series) | Data: general tabular (rows = entities; columns = heterogeneous features) |
| Task: forecast the next value in a series | Task: predict a target per row / entity |
| Models to beat: time-series foundation models | Models to beat: tabular foundation models (TabPFN, XGBoost/LightGBM) |
| Both tracks: add text + agentic reasoning; run closed-resource (Docker, restricted network — model APIs via audited proxy only, no open internet); leakage-controlled |

---

## What this repository contains

| Path | What it is |
|------|-----------|
| `docs/CONCEPTS.md` | Plain-English explainer for every concept (tabular prediction, NLI, calibration, etc.) |
| `docs/CATEGORIES.md` | The prediction families published here, read off the shipped units |
| `docs/AUTHORING-GUIDE.md` | How a unit is laid out, field by field, and who reads each field |
| `units/t4-EXAMPLE-eps-beat/` | A complete worked example: task file, cross-section table, corpus, example answer |
| `units/` (ten more) | Development practice units — real corpora, real task shapes, retired from the held-out set; no outcomes ship, and the organizers score your answers on the Development leaderboard |
| `qfbench2_track_analysis/scoring.py` | The reference scorer, for all three target types (`scoring/scoring.py` is a back-compat shim that re-exports it) |
| `faithfulness/judge.py` | The NLI faithfulness judge you can run locally before submitting |
| `baselines/` | A runnable stdlib RAG agent, a BM25 + house-model scaffold, and an optional citation rail |
| `templates/answer.example.json` | The authoritative `answer.json` shape |
| `templates/{card.toml,task.json,manifest.json}` | Annotated examples of a unit's three non-corpus files (illustrative; the shipped exemplar is authoritative) |

> **The published units are format exemplars, not a representative sample.** The held-out set is
> substantially larger and spans many more task families, across all three target types. Most of
> its families have no published counterpart at all, so do not tune to the shapes you can see
> here — and do not treat the eleven published units as a syllabus. What is guaranteed to carry
> over is the part that is identical everywhere: the submission contract, the answer schema, and
> the scoring. Treat a corpus document's flat `text` field as the normal case.

---

## Submission format

Your submission is a **Docker image** that implements the `analyze` command:

```bash
analyze \
  --task   /input/task.json \
  --corpus /input/corpus \
  --out    /output/answer.json
```

The harness runs your image as `docker run <image> analyze --task … --corpus … --out …`, so
**`analyze` arrives as the first argument, not as part of the image's own configuration.** Your
image must either expose `analyze` as an executable on `PATH` (build with no `ENTRYPOINT`), or —
if you use `ENTRYPOINT ["python", "analyze.py"]` — have `analyze.py` accept `analyze` as a
leading positional argument. An image that ignores the verb exits 2 on every unit before reading
any input. `baselines/baseline_agent/cli.py` shows the second shape; the full contract is
[`SUBMISSION_CLI.md`](SUBMISSION_CLI.md).

### answer.json — what the output must look like

For a **classification** task (e.g., EPS beat/miss/inline):

```json
{
  "task_id":      "t4-2024q2-eps-aapl-001",
  "schema_version": "3",
  "target_type":  "classification",
  "entity_predictions": [
    {
      "entity_id":  "AAPL",
      "label":      "beat",
      "point_forecast": 1.62,
      "interval":   { "level": 0.90, "lo": 1.48, "hi": 1.78 },
      "claims": [
        {
          "doc_id":     "EDGAR_0000320193_10Q_20240202",
          "span_start": 295,
          "span_end":   627,
          "claim":      "Services revenue grew 11% year-over-year in Q1 FY2024, supporting continued margin expansion."
        }
      ]
    }
  ],
  "evidence_trace": "Retrieved 8 documents. Top span: EDGAR_0000320193_10Q_20240202 295-627 (NLI 0.91). No post-cutoff citations. Embargo: PASS."
}
```

> The authoritative shape is `templates/answer.example.json`; validate against
> `analysis.schema.json` rather than against this excerpt.

For a **regression** task (e.g., credit-event probability), `point_forecast` is the primary
prediction; `label` may be omitted or set to a threshold-derived class. For a **ranking** task,
`point_forecast` is *also* the primary prediction — the ordering is taken from it, and `label` is
**not** read. Put the predicted metric value in `point_forecast`; the optional integer `rank`
field records your ordering for readers but does not feed the score, and if you supply it at all
it must be a permutation of 1..n over the full entity roster.

The common mistake, measured: putting the **rank integer** in `point_forecast` (1 = highest)
inverts the ordering against a metric where larger is better, and scores
`predictive_quality = 0.0`, composite `0.15`, against `1.0` / `0.85` for the metric values
themselves (with an interval equal to the naive rule's). Omitting `point_forecast` on a ranking unit is refused outright. In every case there is one entry in
`entity_predictions` per entity row.

Use the executable `analysis.schema.json` bundled with `qfbench2-common` for required fields
and the [answer template](templates/answer.example.json) for an example. The schema-validation
command under **Quick start** reads that installed schema directly. The optional fields
`schema_version`, `target_type` and `evidence_trace` are not required merely because this README's
example includes them. If supplied, `target_type` must match the unit's declared target type;
a mismatch fails the whole unit. The scorer also requires exactly the task's entity roster and
the prediction appropriate to its target type, as described above.

Fields the schema does not name are allowed in `answer.json`, at the top level and in each
entity row (the template's `_comment` is one); they do not fail the unit. Assume the leakage
scan reads them like every other byte of your output. This differs from `submission.json`,
whose schema refuses unknown keys.

**These are not partial-credit penalties.** A missing `interval.lo` or `interval.hi`, an empty or
absent `claims` array, or an `interval.level` other than the card's — on *any* single entity row —
fails `g1_schema` for the **whole unit**: the unit is scored `t4.schema_invalid`
(`SCHEMA_INVALID_OUTPUT`) at the worst-case `W = 0.0` (scorer 3.1.0: `-0.27`; 0.0 shows as -0.27 on the leaderboard, where
leaderboard = -0.27 + 1.27 × analysis), and no coverage or faithfulness number is
computed at all. Verified by running the scorer on each case.

### Runtime constraints

| Constraint | Rule |
|-----------|------|
| Network | `network = "restricted"` — no open internet. Egress only through the organizer's audited proxy to the organizer-hosted open-model endpoint (`$MODEL_ENDPOINT`, OpenAI-compatible, free within a per-run budget) and **nothing else**; vendor model APIs (api.anthropic.com, api.openai.com, generativelanguage.googleapis.com, any other) are **refused by the proxy** (policy 2026-08-04), and no participant API keys exist. Every connection is logged (domain, bytes, timestamps). |
| Corpus path | `/input/corpus` is read-only |
| Output | Must write `/output/answer.json` before process exits |
| Exit code | Must exit 0 on success |
| Timeout | 10 minutes per unit, set authoritatively in `card.toml [agent].timeout_sec = 600.0` — exceeding this budget causes a g2 timeout failure |
| Image size | Recommended ≤ 15 GB; over 20 GB may be rejected |

The image-size row remains the published recommendation and rejection policy; it is not a
verified automatically enforced image-size quota. The image-layer limit is a different resource.
See the [image submission guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/v2.5.1/docs/IMAGE-SUBMISSIONS.md)
for anonymous public pulls and organizer-confirmed private mirrors.

Write `answer.json` as UTF-8 without a byte-order mark. A byte-order mark or non-UTF-8 bytes
fail: the scorer cannot read the file as JSON, and the unit gets the worst-case score. Only
`/output/answer.json` is scored: an answer written anywhere else, even `/output/<dir>/answer.json`,
counts as no answer. Other files in `/output` are not scored, but under the organizers'
platform rules the output checker reads the whole tree after your process exits, and the unit
scores `no_output` if the tree has any of these:

- more than 256 files, or folders nested 8 or more levels deep;
- a symbolic or hard link, or a special file;
- a file with a setuid, setgid or sticky bit;
- a file more than 64 times larger than the disk space it occupies (a heavily sparse file), two names that differ only in letter case or Unicode form, or a name with a backslash or a control character;
- no files at all, or more than 64 MiB in total in Development (a single file is capped at 64 MiB).

In the Final, a canary string anywhere in the output is scored as contamination.

For CPU, memory and GPU settings, read the unit card and the
[Development runtime guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/v2.5.1/docs/DEVELOPMENT-RUNTIME.md).
The `api` category does not remove a card's GPU grant for permitted local code or authorize
an additional model server. The unit clock includes container creation and any required pull;
the ingestion stage has a separate 12-hour clock across sequential units, and scoring has its
own stage clock. The planned House timing release activates each unit once when the organizer begins that unit's execution setup. Queue waiting and earlier units do not spend that unit's own
window; setup/provisioning and container creation/execution after activation can. Its fixed end
is capped by the card/fallback unit ceiling and the remaining actual ingestion-stage time.
Restarting or retrying under the same allocation resets neither the window nor request counters.
Deployment and verification remain required before opening; no compute allowance grows. These
Development settings do not certify Final resources or announce participant access.

**Why restricted, not open.** The corpus is frozen. The embargo rule forbids fetching documents
or data published after `cutoff_date`. Open internet access at inference time would make the
embargo unenforceable — so the only permitted egress is model-API traffic through the audited
proxy, where every connection is logged and becomes the audit artifact for verification within the joint Final + Verification phase. Vendor-side tools (web search, code execution, retrieval) **must be disabled** in API
calls; this is enforced by rule and audit. Data/text cutoffs are unchanged and still enforced by
the harness (`g2`).

**One mode, one contract.** `category = "api"`: your agent calls the organizer-hosted model
endpoint (`$MODEL_ENDPOINT`); your contribution is the prompts, harness, system prompts, agents
and permitted local numerical artifacts. No participant API keys are injected and none exist
(policy 2026-08-04) — the house endpoint is the only reachable model. **Bring-your-own models
and adapters are not part of this competition** (since 2026-09-18): the former
`byo-large` / `byo-small` categories are invalid since toolkit 2.4.3, and an upload that still
carries one is held by the organizer's intake and never run.

At scoring time the container sees: `HTTP_PROXY`/`HTTPS_PROXY` pointing at the audited proxy,
`MODEL_ENDPOINT` set to the origin of the organizer-hosted House route (e.g. `http://model:8443`,
no path — the OpenAI-compatible API is served under `/v1`), `MODEL_TOKEN` carrying the per-unit
bearer, and `QFBENCH_NETWORK=restricted`; see [Calling the House route](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/docs/HOUSE-MODEL.md#calling-the-house-route). Local smoke runs
without the eval network fall back to `--network=none`, so your agent must degrade gracefully
(still emit a schema-valid `answer.json`) when model APIs are unreachable.

**Local numerical artifacts.** The [Track 4 artifact policy](docs/ARTIFACT-POLICY.md) defines permitted non-neural models, calibration parameters and corpus-only retrieval assets, with disclosure and cutoff requirements. It does not authorize additional neural checkpoints.

**Offline training.** The [Track 4 training policy](docs/TRAINING-POLICY.md) permits eligible
external training data within the existing artifact categories, requires cutoff-aware fitting,
selection and calibration, and defines the narrow exception for approved Nemotron base
pretraining. Evaluation inputs and citations stay within the official task and frozen corpus.

**Reproducibility.** Model versions must be pinned (dated snapshots), the training cutoff of
every model must be disclosed in submission metadata, and temperature/seed pinned where the API
supports it. Entries are verified statistically (bootstrap-CI overlap on rerun).

**House API allocation.** The model budget is **requests per unit**: **25 admitted requests per
unit**, with **at most 4,000 output tokens per request**, both counted by the House route. There
is no per-unit token allowance — the earlier figure of 1,000,000 input plus 100,000 output tokens
per unit is withdrawn and nothing replaces it. See the
[model-API rules](SUBMISSION_CLI.md#rules-for-model-api-use-restricted-mode) for the accounting of
failed or retried requests. Platform availability and deployed enforcement will be
announced separately.

**Leaderboard.** One board; every entry is tagged with its category, models used (pinned
versions), and training cutoffs.

### Minimal Dockerfile pattern

```dockerfile
FROM python:3.13-slim
LABEL qfbench2.interface_version="2.0"   # required on every submission image (gate g0)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# The harness appends: analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json
# so analyze.py receives "analyze" as argv[1] and must declare it, e.g.
#   parser.add_argument("verb", nargs="?", default="analyze", choices=["analyze"])
ENTRYPOINT ["python", "analyze.py"]
CMD ["analyze", "--help"]
```

See `baselines/Dockerfile` for a working reference. The harness interpreter is **Python 3.13** —
pinned as part of the hardware contract published before the **2026-08-10 compute-caps freeze**, and
a ceiling rather than a floor (`nemoguardrails` and `nvidia-nat` both pin `<3.14`).

---

## Installing the shared toolkit

Track 4 inherits scoring utilities from the shared toolkit repository. Install them with:

```bash
# Pin toolkit v2.5.1 for the current submission commands and fixtures.
# The installed package reports version 2.5.1.
pip install "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.5.1#subdirectory=common"
```

> **Pin a tag, never a branch.** Installing from a moving ref means your local result and your
> scored result can diverge without either being wrong.

This gives you the `HierarchicalVerifier`, `F.citation_faithfulness`, `F.embargo_violations`,
and `F.analysis_composite` functions used by `scoring/scoring.py`.

---

## How the NLI faithfulness smoke check works

Before your submission reaches the official scorer, you can pre-check it locally:

```bash
pip install transformers torch   # the judge's NLI models; not needed for the scaffolds or --mock
python faithfulness/judge.py --answer /tmp/my_answer.json --unit units/<unit-id>
```

Without `transformers` and `torch` the command prints `Judge unavailable; no prediction was
scored` and `GATE: FAIL (judge not available — install dependencies)` — that is the missing
dependency, not your answer. The first run downloads the two pinned DeBERTa NLI models.

`--unit` is **required** with `--answer`, and it is the unit *directory* — the one holding
`task.json`, `card.toml`, `manifest.json` and `corpus/` — not the corpus directory alone. All
four are inputs to the gate this is previewing: the roster and the target schema come from
`task.json`, the thresholds from `card.toml`, and a `doc_id` resolves only to a document
`manifest.json` declares. Without them the check would have nothing to score against and would
fail closed.

The check runs the same DeBERTa NLI ensemble and the same public scoring functions the official
scorer uses. From scorer 5.2.0 faithfulness is a **per-claim penalty**, not a pass/fail gate: the
unit's score is multiplied by `1 - F / (F + min(T, 3 × E))` (F false claims, T other claims, E
entities): each false claim costs a share of the unit, other claims beyond 3 × E in total do not dilute
that cost, and a unit with no false claims is not penalised. Per **claim** the check:

1. Aligns your `entity_predictions[]` against the unit's entity roster. A missing, duplicated or
   unknown entity fails here, exactly as it fails the scorer, and refuses the unit.
2. Resolves each cited `doc_id` through the unit's manifest and checks its `doc_date` against the
   cutoff. An unresolved, undated or post-cutoff citation refuses the unit.
3. Marks the claim **false** if it cites a document the manifest does not label with the citing
   entity (or mark `shared`), or a slice outside the document (a `"task"` citation outside the
   citing entity's row of the task table counts as the wrong entity), or if its text is empty,
   over 4000 characters or over 400 judge tokens.
4. Applies the figure check: **every** figure in the claim must appear in a span it cites, read
   against the whole cited span. A claim with a figure no cited span carries is **false** and is
   not put to the judge. A claim citing a span over 8,000 characters is **false**, and so is a
   content-free claim (no figure, only evidence/meta words with a filler word about the evidence,
   or nothing but function words); your own scored values and numbers
   inside the unit's own entity names or tickers are exempt; a word-for-word quote of a cited span
   within the cap passes whole. A claim cites one span of one document, so figures from two
   documents need two claims, one per document; figures from two passages of one document fit
   in one claim only if its span covers both, within the cap.
5. Asks the judge for the probability that each cited passage **contradicts your `claim` text**
   (three-way, averaged over both models). Above `contradiction_bar` = 0.9 the claim is
   **false**; otherwise it is neutral. A word-for-word quote of a cited span is not put to the
   judge.

**The hypothesis is your claim; the premise is the passage you cited.** Quote it, paraphrase it,
or state what it shows, with the figures it carries. A claim that is neutral is never charged and
earns nothing here; whether your evidence supports your forecast is scored by reasoning grading.

Output example (the exemplar unit, one entity, one claim; the probability is illustrative):
```
Scoring 1 claim(s) for 1 roster entit(y/ies) from '/tmp/my_answer.json' against unit 't4-EXAMPLE-eps-beat'
------------------------------------------------------------
  AAPL: P(contradiction) 0.0021 [neutral] ok
    claim: 'Services net sales rose to $23,117 million in the first quarter of fiscal 2024 from $20,766 million a year earlier.'

False claims: 0 of 1; faithfulness factor (multiplies the unit's score): 1.0000
GATE: PASS (admitted; faithfulness factor 1.0000)
```

With the served judge backend (`T4_JUDGE_BACKEND=served`), which returns only two-way
entailment, the contradiction check cannot run: the check prints "Contradiction check: NOT
APPLIED", checks only the other four reasons, and the preview is not rankable. Use the local
ensemble for the full check.

Each false claim is listed with its reasons (`wrong_entity`, `out_of_range`, `malformed`,
`unanchored`, `contradicted`). Fix the citation or the claim, or drop the claim, before
submitting: each false claim costs a share of the unit, at least 1/(F + 3 × E) of it (F false claims,
E entities; for example one false claim among twenty costs 5% on a 7-entity unit and 25% on a
1-entity unit, and padding beyond 3 × E claims does not shrink that share), and a unit whose every
claim is false scores 0.

**The check establishes that your claims are about the right entity, cite real passages, carry
the passages' figures and are not contradicted by them**, not that the prediction was derived
from them. The unit's `card.toml` still carries `faithfulness_threshold = 0.80`: from 5.2.0 that
value is read as "use the per-claim penalty" and nothing else. `penalty_k` = 1 and
`contradiction_bar` = 0.9 are fixed scorer constants, the same for every unit and every entrant; a
card or plan that names either is refused. See `docs/CONCEPTS.md`, "Faithfulness", which also
defines the task-table citation (`"doc_id": "task"`).

**Two NLI quantities.** [`DeBERTaNLIJudge.entail`](faithfulness/judge.py) returns the two-way
entailment-versus-contradiction normalization of the single-candidate call (neutral omitted);
it is unchanged and now feeds only the recorded `prediction_relevance` diagnostic.
`DeBERTaNLIJudge.contradiction` reads the same forward pass as a three-way softmax (entailment,
neutral, contradiction) and returns the contradiction probability; the ensemble averages it over
the two models. Neither is a calibrated probability on financial text.

---

## The tabular prediction baselines

`baselines/` ships **two runnable agents** and one specification (see
`baselines/README.md` for the full release status, including the optional citation rail):

1. **Minimal RAG baseline (`baseline_agent/`, shipped & runnable)** — embargo-aware lexical
   retrieval + a rule-based EPS classifier + a fixed-band interval. Pure standard library, so it
   needs no network at all (it runs even under a local `--network=none` smoke run) with no model
   weights and emits a schema-valid `answer.json`.
   This is the floor a real agent should beat, and the agent the commands below invoke.
2. **TabPFN + gradient-boosting (text-blind) — specification only, not yet released.** The
   text-blind tabular floor; described in `baselines/README.md` but no code is shipped.
3. **Strong retrieval-augmented LLM-over-rows (`strong_rag_baseline/`, scaffold shipped)** —
   runnable with `--mock` or against any local OpenAI-compatible server via `$MODEL_ENDPOINT`.
   BM25 span-chunk retrieval with exact-span citation grounding. It deviates from the original
   sketch: retrieval is lexical only — no dense index and no calibration head — because the
   restricted evaluation network cannot fetch embedding weights. Quality acceptance against
   `baseline_agent/` waits on the staging endpoint; see `baselines/README.md`.

To run the shipped minimal baseline on the worked example, from the root of this repository:

```bash
pip install -r baselines/requirements.txt
python baselines/baseline_agent.py \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/baseline_answer.json
```

---

## Scoring formula

Scorer 5.2.2. For a unit that passes the structural checks (schema, roster, citations resolved
and dated on or before the cutoff):

```
composite = w_acc × predictive_quality + w_cal × interval_quality
score     = composite × (1 − F / (F + min(T, 3 × E)))
            F = false claims, T = other claims, E = entities in the unit; no false claims -> × 1
```

Default weights: `w_acc = 0.70`, `w_cal = 0.30`; `interval_level = 0.90`. The faithfulness
factor is described above and in `docs/CONCEPTS.md`, "Faithfulness". `interval_quality =
naive_IS / (naive_IS + IS)` compares your mean interval score with the unit's declared naive
interval (0.5 = as good as the naive rule's; see "Interval calibration" below). From scorer
5.2.1 the interval part can score above 0.5 only as far as the point forecast beats the naive rule:
`interval_quality` is capped at the larger of 0.5 and `predictive_quality`. The
`predictive_quality` component depends on the task's target type -- declared as `target.type` in
`task.json`, and mirrored as `target_type` under `[scoring.params]` in `card.toml`:

- **classification**: accuracy (fraction of rows where predicted `label` matches ground truth),
  **anchored to the naive rule**.
- **regression**: a ratio against the unit's **declared naive rule** (`reference/naive_answer.json`)
  — `naive_MAE / (naive_MAE + MAE)`, both mean absolute errors over the roster. An exact answer
  scores 1.0, an answer with the naive rule's error scores 0.5, and a worse answer falls toward 0
  as its error grows; the score never goes negative.
- **ranking**: Spearman rank correlation **rescaled to [0, 1]** as `(rho + 1) / 2`, **anchored to
  the naive rule**.

**Anchored to the naive rule** (classification and ranking, from 5.2.0): the raw quality is
mapped so that 0 stays 0, the unit's declared naive rule (`reference/naive_answer.json`) scores
**0.5**, and a perfect answer scores 1, linearly in between on each side. The anchor is the
stronger of the naive rule's own quality and, on a ranking unit, a constant forecast's 0.5.

On every target type, a missing entity or required prediction, or a NaN or Inf in any number
the schema names (`point_forecast`, the `interval` numbers, `rank`, and span offsets), fails
validation for the whole unit before predictive quality is computed. The unit receives the
committed worst-case value; there is no per-row partial-credit replacement and no row is dropped
to shrink a denominator. See
[`align_predictions`](qfbench2_track_analysis/alignment.py) and
[`score_unit`](qfbench2_track_analysis/scoring.py).

A unit whose resolved outcome has **no numeric target** (a pure-label task, e.g. "which action
does the company take on its guidance"), or a classification unit that declares
`interval_leg = false`, has no interval leg: from 5.2.0 its composite is the (anchored) prediction
leg alone, `composite = predictive_quality`, not capped at `w_acc`. When a prompt asks for a
numeric quantity, put that numeric — in the prompt's units — in `point_forecast` / `interval`; an
interval on a probability never covers a dollar `y`.

Participant failures (a structural error, an embargo violation) receive the committed
worst-case unit score `W = 0.0` (shown as -0.27 on the leaderboard, where leaderboard =
-0.27 + 1.27 × analysis) and remain in the evaluation denominator. Faithfulness never
refuses a unit; it multiplies the score. This differs from an organizer fault, which aborts
scoring instead of assigning a participant score. A local
smoke check without resolved outcomes also returns no numerical score and is explicitly
non-rankable; it checks the interface, not prediction accuracy or production faithfulness.

---

## Quick start in six commands

```bash
# 1. Install.
# baselines/requirements.txt is comments only -- the minimal baseline is standard library
# by design -- so this line installs nothing. It is here because steps 4 to 6 need the
# shared toolkit, which brings jsonschema with it.
pip install "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.5.1#subdirectory=common"

# 2. Run the RAG baseline
python baselines/baseline_agent.py \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/answer.json

# 3. Check faithfulness. Exits non-zero when the gate fails.
# Scoring the claims needs the NLI judge: pip install transformers torch
python faithfulness/judge.py --answer /tmp/answer.json \
  --unit units/t4-EXAMPLE-eps-beat

# 4. Validate the answer against the published schema.
# The schema ships inside the toolkit; there is no copy in this repository.
python - <<'PY'
import json, importlib.resources as res, jsonschema
schema = json.loads((res.files("qfbench2_common") / "schemas" / "analysis.schema.json").read_text())
jsonschema.validate(json.load(open("/tmp/answer.json")), schema)   # raises on any violation
print("schema ok")
PY

# 5. Run the scorer's structural and claim rules (no NLI models needed), from the repo root.
# The schema alone accepts a NaN and a label outside the task's vocabulary; this check refuses
# them as the scorer does. A `unit_refused` finding means the whole unit would fail.
python - <<'PY'
import json
from baselines.guardrails_example.citation_rail import check_claim_rules
for f in check_claim_rules(json.load(open("/tmp/answer.json")), "units/t4-EXAMPLE-eps-beat"):
    print(f.code, f.claim_index, f.message)
PY

# 6. Smoke-test via the harness.
# PYTHONPATH is required: the harness imports this repo's qfbench2_track_analysis package,
# and the repo is not pip-installable from a checkout.
#    `--profile smoke` is the default and runs the NON-RANKABLE preview factory
#    (`build_smoke_verifier`); `--profile production` runs the rankable one, which needs the
#    pinned NLI judge and refuses without it. The factory that ran is printed with the verdict.
PYTHONPATH=$PWD qfbench2-smoke units/t4-EXAMPLE-eps-beat /tmp --track analysis
```

---

## Development tips

**Tabular features first.** Start by building a text-blind tabular model on the feature columns.
Its accuracy is your floor. Then add retrieval and check your claims with the local faithfulness
preview before worrying about composite score.

**Watch the faithfulness penalty.** Each false claim costs a share of the unit, and adding more
than 3 × E claims in total (E entities) does not dilute it. Keep claims
extractive: every figure in a claim must be in a span it cites, and computed figures go in
`submitted_reasons`. Run `faithfulness/judge.py --answer … --unit …` frequently — do not leave it
as a final check. A `submitted_reasons` block that does not match the schema (an empty list, more
than 3 reasons, or a reason missing a required field) makes the whole answer invalid, like any
schema error, so run the local checker (`check_submitted_reasons` in
`baselines/guardrails_example/citation_rail.py`) first; leaving reasons out never costs anything.
The reasons contract is in `SUBMISSION_CLI.md`, "How reasoning is scored".

**Stale-filing detection.** Apply a strict `doc_date <= cutoff_date` filter in every retrieval
call. Do not rely on post-processing to discard stale citations — the stale information may have
already affected your reasoning.

**Interval calibration.** The interval leg is an `interval_score` ratio against
the unit's declared naive interval: per row, the width `hi - lo` plus `2/alpha` (20 at 90%) times
the distance by which the truth falls outside `[lo, hi]`, averaged over the roster, and scored as
`naive / (naive + yours)` — 0.5 means as good as the naive rule's interval. Width now costs, and so
does a miss. Measured on a three-entity unit, everything identical except the interval (naive band
`[0.5, 3.5]`): the naive band itself scored **+0.383**, `[1.4, 1.9]` (missing all three values)
**+0.297**, and `[-1e9, 1e9]` **+0.233** — the trivially wide interval is now the worst of the three.
(Scorer 3.1.0 scored only the coverage gap `|interval_coverage - 0.90|` and charged nothing
for width: widening was rewarded until coverage reached 90%, and past 90% it cost at most 0.03 at
the default weights.) Aim for the narrowest interval that still contains the value. The interval
part can score above 0.5 only as far as the point forecast beats the naive rule, so narrowing the
band around the naive rule's own points earns nothing above the naive rule. The learned
calibration head is **not shipped** (see `baselines/README.md`); the minimal
baseline emits a fixed-band interval, which is a floor to beat, not a starting point to tune.

**Firewall.** Your agent runs on a restricted network: no open internet, egress only through the
organizer's audited proxy to the organizer-hosted `$MODEL_ENDPOINT` and nothing else —
vendor model APIs are refused. Retrieval indices, dependencies, and other permitted resources
must be baked into the Docker image or available from the read-only corpus
mount. Vendor-side tools (web search, code execution, retrieval) must be disabled in API calls.
Bring-your-own models and adapters are not part of this competition (ruling of 2026-09-18) —
see [submission categories](SUBMISSION_CLI.md#submission-categories-agent-tracks-only).
Test locally with `docker run --network=none` before submitting to confirm your agent has no
open-internet dependency and degrades gracefully when model APIs are unreachable.

## Competition schedule and submission limits

Development runs through **October 12, 2026**. The joint **Final + Verification phase runs
October 13–25, 2026**. Each team makes **one final submission per track**; organizers perform
verification within that same phase, with no separate participant Verification submission.
If two Final submissions finish this track with the same ranking score, the tie is broken in
favour of the one uploaded earlier.
Registration and Development close together on October 12, 2026 at **23:59 Anywhere on Earth (AoE, UTC−12)**. The joint Final + Verification phase closes on October 25, 2026 at **23:59 AoE**. Other competition dates and task/data cutoffs are unchanged.

At the participant Development opening, Track 4 allows **5 uploads per team per day**
and **20 total uploads per team for this track during Development**. Use your team's single
designated CodaBench account. Local validation and packaging use no attempts; held or cancelled
uploads still count. An upload the platform marks `Failed` does not consume an attempt — the platform's
daily count excludes it. See [submission limits](SUBMISSION_CLI.md#development-submission-limits).

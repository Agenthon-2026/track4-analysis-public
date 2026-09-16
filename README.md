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
supports the claim — the submission is ineligible regardless of how accurate its predictions were.

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
| `units/` (ten more) | Practice units — real corpora, real task shapes, no resolved outcomes; not scored |
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
`predictive_quality = 0.0`, composite `-0.03`, against `1.0` / `0.67` for the metric values
themselves. Omitting `point_forecast` on a ranking unit is refused outright. In every case there is one entry in
`entity_predictions` per entity row.

Use the executable `analysis.schema.json` bundled with `qfbench2-common` for required fields
and the [answer template](templates/answer.example.json) for an example. The schema-validation
command under **Quick start** reads that installed schema directly. The optional fields
`schema_version`, `target_type` and `evidence_trace` are not required merely because this README's
example includes them. If supplied, `target_type` must match the unit's declared target type;
a mismatch fails the whole unit. The scorer also requires exactly the task's entity roster and
the prediction appropriate to its target type, as described above.

**These are not partial-credit penalties.** A missing `interval.lo` or `interval.hi`, an empty or
absent `claims` array, or an `interval.level` other than the card's — on *any* single entity row —
fails `g1_schema` for the **whole unit**: the unit is scored `t4.schema_invalid`
(`SCHEMA_INVALID_OUTPUT`) at the worst-case `W = -0.27`, and no coverage or faithfulness number is
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

**Why restricted, not open.** The corpus is frozen. The embargo rule forbids fetching documents
or data published after `cutoff_date`. Open internet access at inference time would make the
embargo unenforceable — so the only permitted egress is model-API traffic through the audited
proxy, where every connection is logged and becomes the audit artifact for the verification
phase. Vendor-side tools (web search, code execution, retrieval) **must be disabled** in API
calls; this is enforced by rule and audit. Data/text cutoffs are unchanged and still enforced by
the harness (`g2`).

**Two modes, one contract.**

1. **API mode** (`category = "api"`): your agent calls the organizer-hosted model endpoint
   (`$MODEL_ENDPOINT`); your contribution is the prompts, harness, system prompts, and agents.
   No participant API keys are injected and none exist (policy 2026-08-04) — the house endpoint
   is the only reachable model.
2. **BYO mode** (`category = "byo-large"` or `"byo-small"`, retained as legacy names): ship
   one LoRA adapter, rank ≤ 64, for the organizer-served base model. Full model weights,
   full fine-tuning, and a submission-run model server are not permitted. Your code calls the
   supplied `MODEL_ENDPOINT` with `MODEL_NAME`, which names your adapter for that run.
   See [adapter-only BYO](SUBMISSION_CLI.md#adapter-only-byo) for packaging and serving rules.

The [descriptor guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/starter-packs/track4/SUBMISSION-DESCRIPTOR.md)
also permits a model-free deterministic declaration using legacy `byo-small` and `models: []`
with toolkit tag `v2.4.0` (C5 1.1.0). Do not invent a placeholder model entry.
An agent that calls the house endpoint uses `api` and declares the actual model.

At scoring time the container sees: `HTTP_PROXY`/`HTTPS_PROXY` pointing at the audited proxy,
`MODEL_ENDPOINT` pointing at the organizer-hosted OpenAI-compatible endpoint and `MODEL_NAME`
naming the served model when available, and `QFBENCH_NETWORK=restricted`. Local smoke runs
without the eval network fall back to `--network=none`, so your agent must degrade gracefully
(still emit a schema-valid `answer.json`) when model APIs are unreachable.

**Offline training.** The [Track 4 training policy](docs/TRAINING-POLICY.md) permits eligible
external training data within the existing artifact categories, requires cutoff-aware fitting,
selection and calibration, and defines the narrow exception for approved Nemotron base
pretraining. Evaluation inputs and citations stay within the official task and frozen corpus.

**Reproducibility.** Model versions must be pinned (dated snapshots), the training cutoff of
every model must be disclosed in submission metadata, and temperature/seed pinned where the API
supports it. API-based entries are verified statistically (bootstrap-CI overlap on rerun); BYO
entries bit-reproducibly.

**House API allocation.** See the [model-API rules](SUBMISSION_CLI.md#rules-for-model-api-use-restricted-mode)
for the selected House allowance and pending input/failure/retry details. These House limits
do not define a BYO request limit. Platform availability and deployed enforcement will be
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
# Pin the tag, and pin this one: v2.3.1 rejects a descriptor the evaluation verifier accepts
# (it requires at least one `models` entry; the current contract allows `"models": []`).
# `pip show qfbench2-common` reports 2.3.1 from this tag -- the metadata lags the tag. That is
# cosmetic and expected; the code is the v2.4.0 code.
pip install "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.4.0#subdirectory=common"
```

> **Pin a tag, never a branch.** Installing from a moving ref means your local result and your
> scored result can diverge without either being wrong.

This gives you the `HierarchicalVerifier`, `F.citation_faithfulness`, `F.embargo_violations`,
and `F.analysis_composite` functions used by `scoring/scoring.py`.

---

## How the NLI faithfulness smoke check works

Before your submission reaches the official scorer, you can pre-check it locally:

```bash
python faithfulness/judge.py --answer /tmp/my_answer.json --unit units/<unit-id>
```

`--unit` is **required** with `--answer`, and it is the unit *directory* — the one holding
`task.json`, `card.toml`, `manifest.json` and `corpus/` — not the corpus directory alone. All
four are inputs to the gate this is previewing: the roster and the target schema come from
`task.json`, the thresholds from `card.toml`, and a `doc_id` resolves only to a document
`manifest.json` declares. Without them the check would have nothing to score against and would
fail closed.

The check uses the public DeBERTa NLI ensemble and shared scoring primitives. It is a local
preview: the organizers have [announced pending normalization and calibration changes](https://github.com/Agenthon-2026/track4-analysis-public/issues/1#issuecomment-5534948217),
and production equivalence requires the published judge version and environment. Per
**roster entity**, rather than per sentence you wrote, it:

1. Aligns your `entity_predictions[]` against the unit's entity roster. A missing, duplicated or
   unknown entity fails here, exactly as it fails the gate.
2. Resolves each cited `doc_id` through the unit's manifest and checks its `doc_date` against the
   cutoff. An unresolved, undated or post-cutoff citation is a **violation**, not a shrug.
3. Builds the **canonical hypothesis** for that entity — one sentence derived from the values you
   submitted (`label` / `point_forecast` / `rank` / `interval`) plus the trusted task schema — and
   asks the judge whether a cited span entails *that*.
4. Reports the fraction of roster entities whose prediction is supported (a span entailing above
   `tau_citation`, 0.5).

**The hypothesis is your prediction, never your prose.** Your `claim` text is parsed and shown, but
it is not what the judge is asked about, so accurately describing a passage you cited cannot make a
wrong prediction faithful. Writing more claims cannot help either: the denominator is the roster.

Output example (the exemplar unit, one entity):
```
Scoring 1 prediction(s) from '/tmp/my_answer.json' against unit 't4-EXAMPLE-eps-beat'
(1 participant claim(s) parsed; the faithfulness denominator is the trusted roster: 1)
------------------------------------------------------------
  AAPL: 0.2455 [FAIL]
    hypothesis: 'The eps outcome of Apple Inc. (AAPL) is inline. The 90% prediction interval for the eps outcome of Apple Inc. (AAPL) is 1.35 to 1.65.'

Faithfulness (fraction of supported predictions): 0.0000
GATE: FAIL
```

A faithfulness score below 0.80 means your submission is ineligible. Cite a passage that supports
the prediction — or change the prediction — before submitting.

**The faithfulness threshold is 0.80 for admission.** Each individual citation is checked
against the per-citation NLI threshold of 0.5 (`tau_citation` in `card.toml`): an entity's
prediction is supported if some cited span entails it above 0.5. The admission gate then requires
that at least 80% of the roster's predictions are supported (`faithfulness_threshold = 0.80`).

**The current NLI quantity is two-way.** Each member uses the entailment-versus-contradiction
normalization returned by the single-candidate call in
[`DeBERTaNLIJudge.entail`](faithfulness/judge.py), then the ensemble averages those values.
Neutral is omitted from that normalization; this is not a three-way probability or proof that
the passage is non-neutral. The hypothesis comes from the submitted prediction, as described
above. This release preserves that quantity and the existing scoring formula. A later scoring
change requires a versioned release; this clarification does not establish a completed
calibration or runtime-equivalence result.

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
   `analyze` uses the official `$MODEL_ENDPOINT` and `$MODEL_NAME` by default. With no endpoint,
   or with `--mock`, it runs a deterministic offline fallback. Every task follows the same
   target-aware prediction and exact-span citation path. A local GGUF experiment is available
   through `--local-llama`, disabled by default. Retrieval is BM25 only; a dense index and learned
   calibration head are not shipped. See `baselines/strong_rag_baseline/README.md`.

The container check is `bash baselines/smoke_image.sh /tmp/t4-out`. It requires a working Docker
daemon and fails when container validation cannot run. Its offline result establishes neither
official NLI admission nor predictive quality; public practice units have no resolved outcomes.

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

The composite score for an eligible submission (faithfulness ≥ 0.80, no embargo violations) is:

```
composite = w_acc × predictive_quality − w_cal × |interval_coverage − interval_level|
```

Default weights: `w_acc = 0.70`, `w_cal = 0.30`; `interval_level = 0.90`. The calibration
term is a **penalty** on the gap between empirical coverage and the nominal level (adding raw
coverage would reward trivially wide intervals). The `predictive_quality` component depends on
the task's target type -- declared as `target.type` in `task.json`, and mirrored as
`target_type` under `[scoring.params]` in `card.toml`:

- **classification**: accuracy (fraction of rows where predicted `label` matches ground truth).
- **regression**: MAE skill against the **cross-entity mean of the realized values** —
  `clamp(1 - MAE / baseline_MAE, 0, 1)`. A perfect model scores 1.0, a model no better than that
  mean scores 0.0, and a *worse*-than-baseline model also scores 0.0: the skill score is clamped
  at zero, it does not go negative.
- **ranking**: Spearman rank correlation **rescaled to [0, 1]** as `(rho + 1) / 2`. A perfect
  ordering scores 1.0, a random one about 0.5, a perfectly reversed one 0.0.

On every target type, a missing entity or required prediction, or a nonfinite supplied numeric
value, fails validation for the whole unit before predictive quality is computed. The unit
receives the committed worst-case value; there is no per-row partial-credit replacement and
no row is dropped to shrink a denominator. See
[`align_predictions`](qfbench2_track_analysis/alignment.py) and
[`score_unit`](qfbench2_track_analysis/scoring.py).

`interval_coverage` is the empirical 90% coverage across the question set (fraction of rows
where the true value falls inside `[lo, hi]`). A unit whose resolved outcome has **no numeric
target** (a pure-label task, e.g. "which action does the company take on its guidance") has no
calibration leg: the coverage term is dropped and `composite = w_acc × predictive_quality`
(same 0.7 ceiling as a perfectly calibrated numeric unit). When a prompt asks for a numeric
quantity, put that numeric — in the prompt's units — in `point_forecast` / `interval`; an
interval on a probability never covers a dollar `y`.

Participant failures, including failed faithfulness or embargo checks, receive the committed
worst-case unit score `W = -0.27` and remain in the evaluation denominator. This differs from
an organizer fault, which aborts scoring instead of assigning a participant score. A local
smoke check without resolved outcomes also returns no numerical score and is explicitly
non-rankable; it checks the interface, not prediction accuracy or production faithfulness.

---

## Quick start in five commands

```bash
# 1. Install.
# baselines/requirements.txt is comments only -- the minimal baseline is standard library
# by design -- so this line installs nothing. It is here because step 4 and step 5 need the
# shared toolkit, which brings jsonschema with it.
pip install "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.4.0#subdirectory=common"

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

# 5. Smoke-test via the harness.
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
Its accuracy is your floor. Then add retrieval and verify that faithfulness passes before worrying
about composite score.

**Watch the faithfulness gate.** Fluent prose does not earn it: the judge is asked whether a cited
span entails your *prediction*, not whether it entails the sentence you wrote about it. Cite the
passage that moves the number you are forecasting, and run
`faithfulness/judge.py --answer … --unit …` frequently — do not leave it as a final check.

**Stale-filing detection.** Apply a strict `doc_date <= cutoff_date` filter in every retrieval
call. Do not rely on post-processing to discard stale citations — the stale information may have
already affected your reasoning.

**Interval calibration.** There is **no width or sharpness penalty anywhere in the scorer**, so a
trivially wide 90% CI is not merely tolerated — it is rewarded. Measured on a three-entity unit,
everything identical except the interval: `[1.4, 1.9]` (missing the value) scored **-0.037** with
coverage 0.0, while `[-1e9, 1e9]` scored **+0.203** with coverage 1.0. The only calibration term is
`|interval_coverage - 0.90|`, so what it actually asks for is *empirical coverage near 90%* across
the task set — not narrowness. Intervals that are too wide hurt you only by pushing coverage above
0.90; on any single unit, widening strictly helps. The learned calibration head is **not shipped** (see `baselines/README.md`); the minimal
baseline emits a fixed-band interval, which is a floor to beat, not a starting point to tune.

**Firewall.** Your agent runs on a restricted network: no open internet, egress only through the
organizer's audited proxy to the organizer-hosted `$MODEL_ENDPOINT` and nothing else —
vendor model APIs are refused. Retrieval indices, dependencies, and other permitted resources
must be baked into the Docker image or available from the read-only corpus
mount. Vendor-side tools (web search, code execution, retrieval) must be disabled in API calls.
BYO model submissions follow the [adapter-only contract](SUBMISSION_CLI.md#adapter-only-byo).
Test locally with `docker run --network=none` before submitting to confirm your agent has no
open-internet dependency and degrades gracefully when model APIs are unreachable.

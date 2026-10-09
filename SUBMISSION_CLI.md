# QFBench 2.0 — Published Submission CLI Contract (`interface_version = 2.0`)

A submission is a **Docker image**. The organizer runs it in the sealed scoring environment;
the image must implement the track verb below. The harness invokes the image, the image reads
from a read-only input mount, writes to an output mount, and **exits 0** on success.

```
docker run --rm \
  --network=none|qfb2-eval \             # "none" (simulation) or the internal eval network (agent tracks) — see "Network modes"
  --cpus=<card.cpus> --memory=<card.memory> [--gpus all] \
  -v <unit-dir>:/input:ro \              # read-only inputs — the UNIT DIRECTORY itself is mounted at /input
  -v <run>/output:/output \              # outputs (deliverables + logs); a normal read-write bind
  [-v <run>/output:/app/output] \       # T1 ONLY: the same host dir, also at the QFBench path (see invariant 8)
  <SUBMISSION_IMAGE> <verb> [args]
```

**The verb is the container command.** It arrives as the first argument after the image
reference, so your image must either resolve it from `PATH` (build with no `ENTRYPOINT` — the
Track 3 reference baseline does this, shipping `simulate` and `simulate-batch` as executables)
or consume it as a leading positional (the `ENTRYPOINT ["python", "agent.py"]` pattern, where
`agent.py` declares `parser.add_argument("verb")`). An image that does not accept the verb fails
every unit — as `127` if the verb is not on `PATH`, as `126` if it is present but not executable,
or as whatever your own argument parser exits with if it consumes and rejects it. All three are
recorded as **your** failure, not an organizer fault, and the unit takes the pre-committed worst
value **W = 0.0** (scorer 3.1.0 used −0.27). On the leaderboard it shows as −0.27
(leaderboard = −0.27 + 1.27 × analysis). Measured: an absent `answer.json` is `no_output`
at 0.0, and an empty one is `malformed_output` at 0.0. Zero is the bottom of the `[0, 1]` domain,
so a unit you fail to produce output for scores no better than the worst admissible answer, and
below any admissible answer on a unit with a numeric target.

The harness logs `sha256(image)` (anti-cheat), applies the card's CPU, memory, GPU and network
settings, and mounts only files whose `manifest.json` checksum matches.
`LABEL qfbench2.interface_version="2.0"` is required on the image.

The Final cannot run an image that declares a Docker `VOLUME`, including one inherited from its base
image. Such an upload is marked Failed when its run starts and does not use an attempt; remove the
`VOLUME` (or choose another base image) and upload again.

For Development, Coding and Explainability take the per-unit timeout from `[agent].timeout_sec`;
Forecasting and Simulation use the launcher's 1,800-second fallback where no timeout is supplied.
The unit clock includes container creation and an image pull when needed. The ingestion stage
runs units sequentially within a separate 43,200-second (12-hour) platform clock; scoring has
its own stage clock. In the planned timing release, a House unit activates once when the organizer begins that unit's execution setup. Its fixed end is capped by the card/fallback unit ceiling and
the remaining actual ingestion-stage time. Queue waiting and earlier units do not spend its own
window; setup/provisioning and container creation/execution after activation can. Restarting or
retrying under the same allocation resets neither the window nor request counters. Credentials
last at most 7,200 seconds from issue and never beyond that fixed end. Deployment and verification
remain required before opening; this changes no compute allowance.
See the [Development runtime guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/v2.5.1/docs/DEVELOPMENT-RUNTIME.md)
for applied limits and pending access status. Development settings do not certify Final resources.

Build a `linux/amd64` image identified by its immutable digest. Follow the
[image submission guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/v2.5.1/docs/IMAGE-SUBMISSIONS.md)
for anonymous public pulls and the organizer confirmation required before using a private mirror.
A descriptor category or image-access field does not itself make a service available.

## How an upload is made

An upload is a **zip, not an image reference**. Push your `linux/amd64` image to a registry that
allows anonymous pulls by digest (the image submission guide above), write `submission.json`
with that digest (the sealed descriptor, see the
[descriptor guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/v2.5.1/starter-packs/track4/SUBMISSION-DESCRIPTOR.md)),
then let the toolkit seal and pack it:

```bash
qfbench2 submission pack --descriptor submission.json --team-number <your team number> --out submission.zip
```

`pack` asks for your Team Key on a hidden prompt, derives your `team_id`, and writes
`submission.zip` containing `submission.json` and `team-claim.json` -- the
[team-claim guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/v2.5.1/starter-packs/track4/TEAM-CLAIM.md)
explains the claim and what happens when it is wrong. Upload `submission.zip` on this track's
CodaBench competition page from your team's designated CodaBench account; the page link was
issued to registered teams at the Development opening and is in the participant announcements.
The Team Key never goes into the zip and is never sent to anyone.

## Development submission limits

At the participant Development opening, **Track 4 allows 5 uploads per team per day**,
with **20 total uploads per team for this track during Development**. Upload through your
team's single designated CodaBench account. Held or cancelled uploads count even when they
receive no score; local validation and packaging use no attempts. An upload the platform marks
`Failed` does not consume an attempt — the platform's daily count excludes it. Track 1 has a 1-per-day limit;
Tracks 2, 3 and 4 retain 5 per day.

Development runs through **October 12, 2026**. The joint **Final + Verification phase runs
October 13–25, 2026**. Each team makes **one final submission per track**; organizers perform
verification within that same phase, with no separate participant Verification submission.
Registration and Development close together on October 12, 2026 at **23:59 Anywhere on Earth (AoE, UTC−12)**. The joint Final + Verification phase closes on October 25, 2026 at **23:59 AoE**. Other competition dates and task/data cutoffs are unchanged.
See the [Development submission limits](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/docs/DEVELOPMENT-RUNTIME.md#submission-limits-at-the-development-opening).

## Network modes (per unit card, `[environment].network`)

There are exactly two network modes; every unit card declares one. **There is never open
internet** in official scoring.

| Mode | Who | Meaning |
|---|---|---|
| `none` | **Simulation (T3)** | Fully offline (`--network=none`). Exactly the historical closed-resource behavior; the container has no network, so any outbound connection attempt fails. |
| `restricted` | **Agent tracks (T1 coding, T2 forecasting, T4 Explainability)** | No open internet. Egress **only** through the organizer's audited proxy to the **organizer-hosted model endpoint** given by `MODEL_ENDPOINT` (open models, free to use, per-run budget). Every connection is logged (domain, bytes, timestamps); the log is the audit artifact for verification within the joint Final + Verification phase. |

> ### ⚠️ Agent tracks: there is no third-party model-API access
>
> Read this before you design your agent. (Track 3 is unaffected — it runs fully offline.)
>
> The proxy allowlist contains the organizer-hosted endpoint and **nothing else**. Calls to
> `api.anthropic.com`, `api.openai.com`, `generativelanguage.googleapis.com` or any other vendor
> API **will be refused by the proxy**, and there is no route around it: the eval network is
> `--internal`, so the proxy is the only path off the host.
>
> There is one model access: the **House endpoint** — call `$MODEL_ENDPOINT/v1/chat/completions`
> with `MODEL_NAME` and the `MODEL_TOKEN` bearer (see the environment contract below). Free,
> metered per run. **Bring-your-own models and adapters are not part of this competition**
> (since 2026-09-18): no LoRA adapter path, no in-image model weights path, nothing fetched
> at run time.
>
> **No participant API keys exist.** The harness injects none and there is no mechanism for a
> submission to supply one, so a vendor key would have nothing to reach even if you had one.

Data and text cutoffs (gate `g2_cutoff_resource`) are unchanged and still enforced by the harness
in both modes — network access is for **model calls only**, never for fetching data.

### Submission categories (agent tracks only)

Track 3 (simulation) sits outside these categories: submissions are simulators and the network
stays `none`. For the agent tracks, every submission declares one category in `submission.json`:

| Category | What you bundle | Model access |
|---|---|---|
| `api` | prompts / harness / system-prompts / agents and permitted local numerical artifacts | the **house endpoint only**, via the proxy |

**Every submission runs against the House model.** Submitting your own model or adapter is not
part of this competition, so `api` is the category for every entry on this track (the shipped
model-free baseline declares `api` too). The former `byo-small` / `byo-large` values are invalid
since toolkit 2.4.3: `qfbench2 submission pack` refuses them, and an upload that still carries
one is held by the organizer's intake and never run. The unit card remains the authority for
your container's resource limits.

Offline training and the narrow pretraining cutoff exception are defined in the
[Track 4 training policy](docs/TRAINING-POLICY.md). They do not expand these categories.

**Local numerical artifacts.** The [Track 4 artifact policy](docs/ARTIFACT-POLICY.md) defines permitted non-neural models, calibration parameters and corpus-only retrieval assets, with disclosure and cutoff requirements. It does not authorize additional neural checkpoints, with one exception: the NeMo Retriever embedding models that the [Track 4 starter pack](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/starter-packs/track4/AGENTS.md#accelerated-libraries-on-this-track) recommends, baked into the image at build time.

### Adapter-only BYO

Withdrawn. This section described a LoRA-adapter option; since 2026-09-18 bring-your-own
models and adapters are not part of this competition, and the descriptor no longer accepts the
`byo-*` categories. Every submission runs against the House model through `MODEL_ENDPOINT`.

`gpu = true` on a task card grants a device for permitted local code. The `api` category denotes
House access and does not remove that GPU grant. This does not authorize an additional model
server.

### Container environment contract (`restricted` mode, set by the harness)

| Variable | Value |
|---|---|
| `HTTP_PROXY` / `HTTPS_PROXY` | the audited egress proxy. **Read these from the environment; never hardcode a proxy host** — the address is an operational detail and it has changed. Most HTTP clients honour them automatically |
| `NO_PROXY` | hosts that must bypass the proxy |
| `MODEL_ENDPOINT` | the **origin** of the organizer-hosted House route (`scheme://host:port`, no path). The OpenAI-compatible API is served under `/v1`: `POST $MODEL_ENDPOINT/v1/chat/completions`. `$MODEL_ENDPOINT/chat/completions` (no `/v1`) is refused with 403. This is the **only** model API you can reach |
| `MODEL_NAME` | the organizer-supplied model id for this run — the pinned House model. Use it unchanged in client calls |
| `MODEL_TOKEN` | the per-unit bearer credential. Send `Authorization: Bearer $MODEL_TOKEN` on every request; without it the route answers 401. With the OpenAI client: `OpenAI(base_url=os.environ["MODEL_ENDPOINT"].rstrip("/") + "/v1", api_key=os.environ["MODEL_TOKEN"])`. Full contract: [Calling the House route](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/docs/HOUSE-MODEL.md#calling-the-house-route) |
| `QFBENCH_NETWORK` | `restricted` (or `none` for simulation / local fallback) |

### Rules for model-API use (`restricted` mode)

1. **Vendor-side tools OFF.** Web search, code execution, retrieval, and any other vendor-side
   tool MUST be disabled in every API call. Enforced by rule + audit of the proxy logs.
2. **Pin model versions.** The house endpoint serves the organizer's pinned base. Floating
   aliases (`*-latest`) are not reproducible and are rejected at verification.
3. **Disclose training cutoffs.** The training cutoff of every model used MUST be declared in
   submission metadata (`models[].training_cutoff` in `submission.json`).
4. **Pin temperature/seed** where the API supports it. Entries are verified *statistically*
   (bootstrap-CI overlap on organizer rerun for T2/T3/T4). For T1, what has to match on a rerun
   is the submitted image and program, not the House model's answers: a per-unit verdict that
   differs only because the House model answered differently is not a violation.
5. **House API allocation — the budget is requests per unit.** The House allowance is
   **25 admitted requests per unit**, with **at most 4,000 output tokens per call**; both are
   counted and applied by the House route. Omitted output limits use 4,000; larger limits are
   reduced to 4,000, and smaller valid limits are preserved. Multiple generated alternatives are
   refused. **There is no per-unit token allowance** — the earlier figure of 1,000,000 input plus
   100,000 output tokens per unit is withdrawn and nothing replaces it.
   An admitted request is charged before forwarding: upstream failures or a lost response do
   not refund it. An admitted participant or SDK retry can consume another slot, even with the
   same content. Invalid requests refused before admission do not consume a slot. Budget
   automatic retries. Platform availability and deployment status
   will be announced separately.

**One leaderboard.** All categories rank on a single board; every entry is tagged with its
category, the models used (pinned versions), and their training cutoffs.

| Track | Verb | Inputs (under `/input`) | Required output (under `/output`) |
|---|---|---|---|
| **T1 Coding** | `solve --task-dir /input --out /app/output` | task spec + environment files | task-specified deliverables written to **`/app/output`** (QFBench/Harbor convention); the `checks/` step asserts correctness and writes **both** `reward.txt` and `reward.json` (see T1 note below) |
| **T2 Time-Series Forecasting** | `forecast --panels /input/panels/ --text /input/text/ --asof <YYYY-MM-DD> --out /output/forecast.parquet` | `panels/` — multivariate time-series parquet files; `text/` — time-stamped text corpus (news, FOMC, macro releases); all timestamps must be ≤ `--asof` (gate g2 enforces both) | joint predictive distribution conforming to `forecast.schema.json`; sidecar `forecast_meta.json` required; **`forecast_rationale.md` required and never scored** (see T2 note below) |
| **T3 Simulation** (single scenario) | `simulate --config /input/scenario.json --out /output/trace.parquet` | scenario config + ABIDES environment | message-level trace conforming to `sim_scenario.schema.json` + `events.json` (counts/timing) |
| **T3 Simulation** (batched, family GB) | `simulate-batch --batch-dir /input/scenarios --out-dir /output` | `batch.json` — the sub-scenario roster; `scenarios/` — one config per sub-scenario. These units have **no** top-level `scenario.json`. | one output subdir per sub-scenario, each with `trace.parquet` + `events.json`, plus `batch_events.json` at the root of `--out-dir` |
| **T4 Explainability** | `analyze --task /input/task.json --corpus /input/corpus/ --out /output/answer.json` | `task.json` — tabular dataset (rows = entities) + a `target` block whose `target.type` is classification/regression/ranking; `corpus/` — frozen evidence corpus | prediction + interval + citations per row conforming to `analysis.schema.json`; if present, `target_type` in the output must match the unit's target type (`target.type` in task.json, `target_type` in card.toml) |

> **T2 status (2026-08-20).** The unit-layout half of this is closed: `--panels` names the STAGED
> unit's `panels/` directory. `stage_bundle.py` relocates root panels into `panels/` and its S6
> gate refuses to emit a unit whose `panels/` is empty, and participant containers mount the
> staged tree, never the raw repo. Verified by execution: `--panels /input/` dies with "no
> .parquet found"; `/input/panels/` scores the full chain. The interface half ships in `track2-forecasting-public` — a `forecast`
> CLI, a console-script entry point and a `Dockerfile`, built and run on linux/arm64 (GH200) and
> admitted by g0–g3 against the exemplar unit under `--network=none`.

> **T2 note — `forecast_rationale.md`.** Alongside `forecast.parquet` a submission writes
> `forecast_rationale.md` to `/output`: the derivation behind the distribution. Numbered steps —
> the anchor, each adjustment with its size and what supports it, then the scale and shape —
> naming the series and dates computed from and the documents cited, and ending with an
> adjustment ledger so the arithmetic can be followed.
>
> **It is required and it is never scored.** No submission is ranked higher or lower because of
> this file; `g1_schema` checks only that it exists and is non-empty, and no scoring code reads
> its content. It exists because a submission that recalled its answer and one that derived it
> are indistinguishable as a set of draws, so the parquet alone cannot support any review at all.
>
> It is read as a **screen over the top of the leaderboard**, not a gate: a flag opens a human
> review and cannot by itself produce a DNF or move a score. That restriction stands until a
> false-positive rate has been measured on a large honest corpus — the current measurement is
> 4/4 true positives and 0/4 false positives, and 0-of-4 carries a 95 % interval reaching ~0.6.
> Two alternative mechanisms were tested and rejected: re-executing the trace (backward-built
> traces reproduced *more* exactly than honest ones, so as a gate it favours the cheater) and
> scanning the text for leakage admissions (a guarded prompt drove self-declaration from 100 % to
> 0 % with no change in the numbers). Method and data:
> `track2-forecasting-public/docs/RATIONALE-REVIEW.md`.

**Your image must implement BOTH Track 3 verbs.** The harness picks the verb per unit, from the
unit's contents: a unit carrying `batch.json` + `scenarios/` is dispatched to `simulate-batch`,
everything else to `simulate`. Six of the public dev units (`t3-gbatch-*`) are batched.

**Contract invariants (enforced by gate `g0_integrity` / `g1_schema` / `g2_cutoff_resource`):**

1. The image must honor the card's network mode: `none` (simulation) means fully offline — the
   container has no network, so any outbound connection attempt fails; `restricted` (agent
   tracks) means egress only through the audited proxy to the house model endpoint — a connection
   to anything outside that allowlist is refused, every connection is logged, and no vendor model
   API is on the allowlist.
2. Output must validate against the track output schema *before* any scoring (`g1_schema`).
3. The image must not read any path outside `/input` and `/output`; the canary registry and held-out
   targets are never mounted.
4. Determinism: the harness sets `QFBENCH_SEED`; organizer verification within the joint Final + Verification phase
   reruns on fresh seeds/resamples and compares against the final-submission result (reproducibility gate).
   Because the rerun's `QFBENCH_SEED` is fresh, a seed meant to repeat must be a constant in your code.
   Track 1: what has to match on the rerun is the submitted image and program, not the House model's answers,
   so a per-unit verdict that differs only because the House model answered differently is not a violation.
5. Wall-clock and resource caps are per-track (`card.environment`); exceeding them is a `g2` failure.
6. **T2 text cutoff (g2):** every document in `/input/text/` must have a timestamp field ≤ `--asof`.
   The harness checks text timestamps in addition to panel data timestamps. A document with a
   post-as-of date causes a `shared.leakage.cutoff_violation` failure label
   (`FailureLabel.LEAKAGE_CUTOFF` in `qfbench2_common.failure_labels`).
7. **T4 target type:** the `target.type` field in `/input/task.json` (a nested `target` block, not a flat
   `target_type` key; mirrored as `target_type` under `[scoring.params]` in `card.toml`) declares
   the task as `classification`, `regression`, or `ranking`. The `answer.json` output may include a
   `target_type` field; if it does, it must match. Mixed task types within one unit are not allowed.
8. **T1 deliverable dir + dual reward (QFBench heritage):** Track 1 *is* QFBench, so it inherits
   QFBench's conventions. The agent writes its deliverables to **`/app/output`**, which is what
   `--out` is set to and what the units' `instruction.md` and `checks/test_outputs.py` say. The
   harness binds the run's output directory at **both** `/app/output` and `/output`, so the
   minority of units phrased against a bare `/output` are captured identically — writing to
   either path is safe, and neither is silently discarded.
   `checks/test.sh` runs **offline** via `python -m pytest` and writes **both**
   Harbor's `/logs/verifier/reward.txt` (1/0) **and** `<output>/reward.json` + `pytest_report.json`
   for the Agenthon g0–g3 verifier and DI failure-label overlay. The same unit therefore runs under
   **both** Harbor (`harbor run --path units ...`) and the Agenthon harness
   (`qfbench2 smoke <unit> <out> --track coding`). See
   `track1-coding-public/docs/QFBENCH-HERITAGE.md`. (T2/T3/T4 keep the generic `/output`
   contract above.)


## How faithfulness is scored (Track 4, scorer 5.3.0)

Faithfulness is a **per-claim penalty**, not an admission gate. Each claim in
`entity_predictions[].claims` is either **false** or **neutral**, and each false claim costs a share of the unit. The unit's analysis score is multiplied by
`1 - F / (F + min(T, 3 × E))`, where F is the number of false claims, T the number of other
claims and E the number of entities in the unit. A unit with no false claims is not penalised,
and a unit whose every claim is false scores 0. Other claims dilute the false ones only up to a
cap of 3 × E claims in total (three times the number of entities, counted over the whole unit,
not a limit per entity). Up to the cap the cost is the plain share: on a unit with 7 or more
entities, one false claim among twenty claims costs 5%; with fewer entities the cap is lower,
so it costs more (on a 1-entity unit, one false claim among twenty costs 1/(1 + 3) = 25%). Past
the cap, adding more claims does not shrink what a false claim costs (on a 10-entity unit, one
false claim always costs at least 1/31 of it). Neutral claims are never charged and earn
nothing; beyond the cap they do not change the factor. A content-free claim (no figure, only
evidence or meta words with a filler word about the evidence, or nothing but function words) is not neutral: it is false (below). Nothing
about faithfulness refuses a unit; structural errors still do (schema, a missing, extra or
duplicated entity, a non-finite number, an unresolved, undated or post-cutoff citation, a
malformed citation). These checks read the citations in `claims`. A citation in a reason that
does not resolve, is dated after the cutoff or points outside its document never refuses the
unit; one that breaks the schema (a negative offset, a missing field, a `doc_id` that is not a
string) does, like any schema error.

**Only the first 20 claims about each entity count (from scorer 5.3.0).** For each entity, the scorer checks
and counts only its first 20 claims, in the order they appear in `answer.json`. Further claims about that
entity are ignored, not penalised: they are not checked, not put to the judge, and not counted in F or T.
Faithfulness and the penalty factor are computed over the counted claims only; E is unchanged. The cap is per
entity, not per unit. The answer as a whole is still checked: the schema applies to every claim, and a
citation that does not resolve, is undated or is dated after the cutoff refuses the unit even in an ignored
claim. Put your most important claims first.

**`faithfulness_rubric` in `task.json` is a legacy field; ignore it.** Every unit's `task.json` still
carries a `faithfulness_rubric` text. The public practice units' `card.toml` files carry one too;
the held-out evaluation cards do not. That text describes the
admission gate used before 5.2.0: a claim counted as supported above an NLI score of 0.5, and a
submission was admitted when 80% of its claims were supported. Neither scorer 5.2.0 nor the
reasoning grader reads the field. The rules that apply are the ones in this section and in
`docs/CONCEPTS.md`.

A claim is **false** when it cites a document the unit manifest does not label for its entity (or
mark `shared`); cites offsets outside the document; is empty, over 4000 characters or over 400
judge tokens; cites a span over 8,000 characters; states **any** figure that no span it cites
carries (read against the whole cited span); is content-free (no figure; once the unit's own
entity names and tickers are set aside, nothing but function words and evidence/meta words; and
either nothing but function words is left, or one of them is a filler word about the evidence:
evidence, passage(s), excerpt, pre-cutoff, cutoff, cite(d), citing, retrieved, top-retrieved,
nearest, placeholder, fallback, inference, context(ual), wording, document(s), source(s),
model-entailed; for example "Pre-cutoff evidence selected for the submitted prediction." A
claim whose only words are ordinary finance words such as forecast, quote, support, submitted or
available, "AAPL has no forecast.", is not content-free); or when the NLI ensemble's
three-way probability that the cited passage **contradicts** the claim exceeds `contradiction_bar` = 0.9. A passage longer than the judge's
window (about 500 tokens) is judged on the window that shares the most words with the claim.
From scorer 5.3.0 the Development board scores with the same NLI judge as the Final, contradiction check
included. Run the local checker
(`check_claim_rules` in `baselines/guardrails_example/citation_rail.py`) before you write `answer.json`: it
applies every rule except the contradiction check, which needs the models (`python faithfulness/judge.py
--answer <answer.json> --unit <unit-dir>` runs it; add `--cache-dir <dir>` to say where the two judge models
are, or are downloaded to, when the default `/model-cache` is not on your machine).
Every other claim is neutral: it is never charged, and it earns nothing here. `penalty_k` = 1
(the power the factor is raised to), `contradiction_bar` = 0.9 and the cap of 3 × E
other claims are fixed scorer constants.

**Claims are extractive facts.** State what the cited passage says, with the figures it carries;
**every figure in a claim must appear in a passage the claim cites**. A claim cites one span of
one document (`doc_id`, `span_start`, `span_end`), so figures from two documents need two
claims, one per document. Figures from two passages of one document fit in one claim only if
its span covers both, within the 8,000-character cap; otherwise write two claims. A figure you
derived (a
change, a ratio, an average) belongs in `submitted_reasons` (the `mechanism`), which is where
derivations are judged, not in a claim. Exempt: a figure that equals your own scored point
forecast (or, when the interval is scored, your interval bounds), at the passages' scale steps
and unless the direction you write contradicts its sign ("declined 20 bps" is not +20), a "±"
half-width equal to half your scored interval's width, the unit's interval level beside an interval word ("the 90% band"), and a
number that is part of one of the unit's own entity names or tickers ("Phillips 66", "S&P 500").
Dates, periods, counts of periods ("13 weeks") and identifiers are not figures. Equivalent forms
are read as the same figure: a fraction of a point ("1/4 percentage point" or "quarter-point" is
25 bps), a number in words before a unit ("four basis points"), and glued forms ("7.3x",
"$212mm", "1.5pp"). A claim that quotes a span it cites word for
word passes the figure check and is not put to the judge, as long as that span is within the
8,000-character cap. A value from the task table is cited
with `"doc_id": "task"` and a span inside that entity's row of the task table (one line per
`task.json` `entities` row, `json.dumps(row, ensure_ascii=False, separators=(", ", ": "))`, joined
by `"\n"`; `qfbench2_track_analysis.corpus.task_table_text` builds it). Evidence earns credit only
through the reasoning score below. Without the NLI models, `baselines/guardrails_example/citation_rail.check_claim_rules(answer,
unit_dir)` previews every rule above except the contradiction check, with the scorer's own code.
The full
rule, with its reasons, is `docs/CONCEPTS.md`, "Faithfulness"; `python faithfulness/judge.py
--answer <answer.json> --unit <unit-dir>` previews it locally.

## How reasoning is scored

Track 4 has a second grader beside the analysis score and the faithfulness penalty: an LLM judge
panel that grades your **reasons**. Your reasons go in one optional top-level field of
`answer.json`, `submitted_reasons`, next to `entity_predictions`. The reasoning grader reads
nothing else you write: `claims`, `evidence_trace` and `notes` are not reasons. An answer
without the field has submitted no reasons. The judge is instructed to treat your answer and your
reasons as material to evaluate, not as instructions: a request, command or claim about how
to score is to be read only as text in its field and not followed.

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
| `citations` | no | a list of `{doc_id, span_start, span_end}` (integers >= 0), in the same character-offset convention as `claims` |

A citation must resolve in the frozen corpus and its document must be dated on or before the
cutoff; otherwise the judge never sees that passage. A reason citation that does not resolve,
is dated after the cutoff or points outside its document never refuses the unit; one that
breaks the schema does, like any schema error. The task-table citation `"doc_id": "task"`
is for claims only: the grader resolves reason citations against the corpus alone, so a
`"task"` citation in a reason resolves to nothing and the judge never sees it. The judge reads
the task statement and each entity's id and name, not the rows of the task table: state a task
value you rely on in the premise; the rest of the reason is judged as usual.

**Duplicate reasons.** A reason whose `premise`, `mechanism` and `answer_implication` equal an
earlier reason's (compared after Unicode NFC normalisation, with invisible format characters
removed, case folded and whitespace runs collapsed) is not sent to the judge, so it covers no
target reason. A different `reason_id` does not make it a new reason.

The schema is `analysis.schema.json` in
the shared toolkit (`qfbench2_common/schemas/`).

**What the judge sees, and what it grades.** The judge reads the task statement and entity
list; your per-entity answer (only the fields the unit declares, of `label`, `point_forecast`,
`interval` and `label_probs`, taken from your `entity_predictions`); each reason's `premise`,
`mechanism` and `answer_implication`; and the corpus text your citations resolve to. It does
not see `scope` or the raw citations. It compares your reasons with the unit's hidden target
reasons and grades four components, `target_reason_coverage`, `evidence_grounding`,
`inferential_link` and `answer_consistency`, plus the flags `valid_grounded_premise`,
`has_answer_implication` and `contradiction`, with 5 judge votes per cell. A target reason that
none of yours covers scores 0 against a denominator of all the unit's target reasons, so
submitting fewer reasons never scores higher. From Track 4 scorer 5.2.0 the reasoning score is
a **bonus** on top of the analysis score (final-score/v2):

    final = -0.27 + 1.27 x analysis + 0.25 x reasoning

`analysis` is your 0..1 analysis score after the per-claim faithfulness penalty (it combines
your prediction and, on units that score one, your interval; from scorer 5.2.1 the interval part
can score above 0.5 only as far as the point forecast beats the naive rule), shown on the
old leaderboard scale (`-0.27 + 1.27 x analysis`: 0 shows -0.27, the old worst case, and 1 shows
1.0); `reasoning` is in [0, 1]. The bonus is uncapped, so the maximum is 1.25. A keyed unit with
no judged reasons (missing, none within the caps, or refused for the deny list) adds 0 to the
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

**Caps.** Reasons are checked in the order you submit them. A reason is judged only if every
citation in it is at most 8,000 characters and, together with the reasons already judged, the
reasons stay within 6,500 bytes and their cited evidence within 46,500 bytes. A reason that does
not fit is not judged and scores 0 (every target reason stays in the denominator); later reasons are
still checked. If no reason fits, the unit's reasoning scores 0. Put your strongest reason
first. The 3,000-byte answer cap still applies to the whole unit. Nothing is clipped.

| cap | limit, per unit |
|---|---|
| one citation: `span_end - span_start` | 8,000 characters (cite the passage, not the document) |
| your per-entity answer as the judge reads it (each row's `entity_id` and declared fields) | 3,000 bytes |
| your reasons as the judge reads them (`reason_id`, `premise`, `mechanism`, `answer_implication`) | 6,500 bytes |
| the cited passages as the judge reads them (each resolved citation's text with its `doc_id`, offsets and reason id) | 46,500 bytes |

The last three are counted the way the grader counts what the judge reads: UTF-8 bytes of compact
JSON. Plain ASCII text is one byte per character; a line break, quote or backslash is two (it is
escaped); accented letters, typographic quotes and non-Latin scripts take two to four; a control
character six; a URI in cited text is masked with the same number of `█` (three bytes each), a
URL in your own reason text with the same number of `#` (one byte each); and
every citation adds about 75 bytes of JSON around its text plus its `doc_id` and offsets. In practice: about 6,000 characters of
plain reason text over three reasons, and about 45,000 characters of plain cited text in a few
citations. The three caps add up to the grader's 56,000-byte limit on what the judge reads from
you, so an answer within them never reaches that limit. The local checker below reports each
(`cap_answer_bytes`, `cap_reason_bytes`, `cap_evidence_bytes`).

**Deny list.** The grader refuses a unit's request, and that unit's reasoning scores 0, if the
text you wrote contains any of these, case-insensitively, as a substring: `leaderboard`,
`canary`, `/home/`, `units/`, `reference/`, `outcome.json`, `team_id`, `team name`,
`participant_id`, `participant name`, `submission_id`, `other submission`. `mechanism` and
`answer_implication` are always checked. Exempt: the corpus text your citations resolve to,
and a `premise` that is a verbatim quote of a corpus document: with every URL masked, it has at
least 3 words and, the document's URLs masked the same way, appears in one corpus document. A
premise that adds any word of your own, a bare token such as `units/`, `canary` or `/home/`, and
a quote of one or two words are checked. So do not put file paths or the other listed tokens in
your own words. URLs in reasons are masked, not refused. The deny list still runs on the URL as written, so a
URL containing a listed token (for example a path with `units/`) is refused; so is a `://` with
no scheme letters before it. Disguised URLs are masked too: look-alike colons and slashes
(fullwidth or other Unicode forms), invisible characters inside a URL, a scheme-less `//host`
and a `www.` host; a deny-listed phrase disguised the same way is refused. These are not
URLs and are left as written: a bare host or path (`example.org/a`), `mailto:` and `data:`, an
IP address, a non-breaking space between the slashes, and dot or bracket obfuscation
(`example[.]org`).

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

## Open Division tag

Do **not** add `house_endpoint_only` -- or any key the descriptor schema does not list -- to
`submission.json`: the schema refuses unknown keys, so a submission carrying it is rejected
before it runs. Whether every model call used the house endpoint exclusively is read from
the audited egress-proxy logs during the joint Final + Verification phase; it drives an "Open Division"
display filter of the single leaderboard (never a separate ranking) and needs nothing
from you.

## Executive summary (read this first)

This development tool runs the baseline and records what was actually measured.
Public practice inputs have no resolved outcomes, so their predictive scores remain empty.
Smoke admission does not measure production NLI support.
Private development outcomes are read after prediction and are never staged as agent inputs.
Evidence annotations are inspectable source mentions, not inferred financial relationships.

## Run

Use the same Python environment as the track scorer and shared toolkit.
The [CLI](./__main__.py) lists supported options:

```bash
python -m baselines.evaluation --help
python -m baselines.evaluation \
  --units units \
  --out "$HOME/t4-evaluation/public-baseline" \
  --mode grounded
```

The output directory must be new and outside all worktrees of this public repository.
It contains a Markdown report, a machine-readable report, and per-run answers,
diagnostics, canonical prediction hypotheses, and evidence candidate ledgers.
In model mode, the evidence ledger records the same retrieval strategy and configured budget
used for predictions. In grounded mode it is a separate retrieval diagnostic. Its
`used_for_prediction` field distinguishes these cases; it does not assert semantic support.

Use `--image` to run an already-built baseline image through Docker with no network.
The tool resolves its immutable image ID and compares its runtime source files with the
evaluator checkout before running; rebuild images whose runtime differs. Only declared task and corpus files
are copied into the container. No host data directories or Docker socket are mounted.
Returned artifacts are collected separately; only regular answer and diagnostics files
are imported. Container runtime faults abort evaluation instead of becoming scored failures.
This tests Docker execution, not the organizer's gVisor environment or model service.
The offline container uses conservative local CPU/memory caps. The runner records the
actual Docker configuration, out-of-memory state and process wall time, then checks them
against each unit's frozen card contract. These observations do not establish hardware
equivalence with the official environment.

Before prediction, the runner persists a complete input/seed/resource roster in
`run-plan.json`. Report loading verifies its digest and complete coverage, as well as
persisted result, answer and diagnostics bindings. Input changes during execution abort
the report. Cards without an explicit agent timeout retain unmeasured resource evidence;
a caller's timeout is not a substitute for the official contract. These local integrity
checks detect drift, not deliberate rewriting of all artifacts by their owner.

Without `--image`, the predictor runs in a separate local process with staged inputs.
This prevents accidental truth-path plumbing, but is **not a filesystem security sandbox**.
Use container mode for strict input isolation. Local `--mode model` requires a configured
house endpoint; the [submission contract](../../SUBMISSION_CLI.md) owns its configuration.
Neither endpoint URLs nor credentials nor raw model responses are included in reports.

## Private development data

Keep the roster, original outcomes and run artifacts in a separate private directory or
private repository. Pass its roster with `--manifest` instead of `--units`:

```json
{
  "version": 1,
  "cases": [
    {
      "id": "independent-case-a",
      "unit_dir": "inputs/case-a",
      "split": "test",
      "group": "independent-event-a",
      "truth_path": "truth/case-a.json"
    }
  ]
}
```

Paths resolve relative to the roster. Input units must satisfy the official card and
manifest checks. Outcomes use the existing track scorer's format; the adapter validates
the exact roster and refuses inconsistent reference data. Omitting `truth_path` leaves
prediction metrics unmeasured. A reference defect aborts the evaluation; it is never
silently removed from an average.

[Dataset validation](./dataset.py) rejects duplicated cases, event groups shared across
splits, and overlapping train/calibration/test periods. An earlier split's outcomes must
precede every cutoff in the later split. The tool does not certify data licensing,
historical data vintages or absence of leakage in trained model weights; retain their
provenance independently. The supplied repository tests use invented data only.

## Interpret results

`development_score` uses the installed official scorer. In smoke mode it deliberately
omits the production faithfulness gate and is useful only for local predictive comparisons.
Failed cases with development truth retain the canonical failure score in the denominator.
A missing score on any planned run suppresses the overall mean. Inspect `by_split` to
keep training diagnostics separate from held-out performance.

`official_score` remains null and `rankable` remains false for all local reports.
`nli_faithfulness` remains null under smoke; `lexical_faithfulness` is a diagnostic proxy.
When numeric truth and aligned predictions exist, `numeric_errors` records raw target-scale
absolute and signed errors. These diagnostics keep differences visible when the official
normalized quality clips to zero; they never replace the official scoring function.
The diagnostics also expose the MAE of the realized cross-section mean, the reference used
by regression skill. That mean is known only after resolution and is not an available
forecasting policy. A small raw MAE can still yield zero skill against this reference.
Production mode requires the organizer's configured judge and external outcomes and never
downgrades to smoke. The runner stops before inference when outcomes are undeclared or the
judge cannot be constructed.

Provenance includes the workspace source digest, Git revision and dirty state, Python,
installed scorer identity, and the actual shared-toolkit source digest. A package version
without a verified build stamp is identified as such. The Docker image ID identifies
participant runtime bytes separately from evaluator workspace bytes.

## Historical snapshots and evidence review

The development-only [historical builder](./historical.py) reads an external event specification
and retrieves Treasury, CPI, foreign-exchange and energy observations from ALFRED.
The supported families and series are declared in [`historical.py`](./historical.py).
It requires `curl` for acquisition; cached
snapshots can subsequently be rebuilt offline. Every series is requested separately, and its
returned column must carry the exact requested vintage date. This matters because a multi-series
request can silently return the latest vintage for some columns. Downloaded bytes, source URLs,
vintages and retrieval times are retained outside the public repository.

The specification has `version: 1` and an `events` array. Each event declares `id`, `cutoff`,
`resolution`, and `split` (train, calibration or test). The builder refuses overlapping event
windows and split leakage before downloading. It creates classification, regression and ranking
views of each event, with the same event group: those views are correlated, not additional
independent samples. The default family covers rates only; additional families do not establish
coverage of all hidden tasks or independence between contemporaneous market shocks.
The corpus is a deterministic extract of historical observations, not a financial forecast.
Only prediction inputs go under each unit; future snapshots and outcomes stay outside those units.

For a separate inflation benchmark, set the external specification's `family` to `cpi_mom`
and add `target_month` (an ISO date on the first of the month) to each event. The supported
seasonally adjusted series are declared in [`historical.py`](./historical.py). Each target
is the month-over-month percent change calculated from two index levels in the resolution
vintage; it is **not a certified first-release value**. Both levels use that same vintage,
so a revised denominator is handled consistently. The target month must be absent from
the cutoff snapshot, immediately follow its latest observation, and be the latest month
in the resolution snapshot. Invalid or incomplete events abort the build.

CPI inputs include prior monthly levels and the last available monthly change. Input tables
use observation dates; their document dates identify the snapshot's availability. CPI
components overlap, and the target views share a release event, so neither entity rows nor
views count as independent samples. Keep this domain's development and held-out rosters
separate and retain the same chronological split checks. The default Treasury build and
its existing cache URLs remain unchanged.

The market-return families use the latest common observation available in each requested
vintage. Returns divide the resolution level by the original cutoff reference, so later
revisions cannot change the starting level. Exchange-rate quotation directions remain as
published; yen per dollar is not silently inverted. The builder refuses stale snapshots,
insufficient recent history and resolutions with no observation after cutoff. These are
specified-vintage targets with publication lag, not certified first-release targets.
Private rosters carry the domain alongside the event group. Series metadata, attribution
and reuse links are recorded with fetched snapshots; retain the source-specific license
assessment separately before using a dataset for acceptance. Different domains sharing a
time window may remain dependent and must not inflate independent-event counts.

```bash
python -m baselines.evaluation.historical \
  --spec /private/evaluation/events.json \
  --cache /private/evaluation/alfred-cache \
  --out /private/evaluation/new-benchmark

python -m baselines.evaluation.review \
  --report /private/evaluation/run/report.json \
  --manifest /private/evaluation/new-benchmark/manifest.json \
  --out /private/evaluation/new-review

python -m baselines.evaluation.compare \
  --before /private/evaluation/baseline/report.json \
  --after /private/evaluation/candidate/report.json \
  --out /private/evaluation/comparison.json
```

Use `--units units` instead of `--manifest` to audit public runs. The review packet contains the
canonical hypothesis and resolved citation text for every entity. Its adjacent annotations start
as `unreviewed`; passing date/offset checks never creates semantic approval. Edited annotations
can be passed back with `--annotations` into a new review directory. A completed judgment requires
an attributed reviewer and reason, and annotations bind to the exact packet digest. Automated
or model-assisted judgments must be attributed as such. Historical reports without an original
answer hash explicitly record that limitation; their hypotheses and citation positions are checked.

Comparisons require identical complete case/seed rosters, input and truth digests, split/group
assignments, judge identity and toolkit bytes. Failed units stay in the mean. Bootstrap intervals
resample event groups, averaging correlated views and seeds inside each group. Inspect the test
split separately; the overall result includes every supplied split. Neither a smoke score nor
a small single-domain bootstrap interval establishes production faithfulness or generalization.
One event group may span several domains when they share a shock. The acceptance audit counts
that group once overall and once within each affected domain; adding domains does not multiply
the total independent-event count. Each case must retain its group, domain and target type
across repeated seeds.

Toolkit tag `v2.4.0` still reports package version `2.3.1`; use the recorded source digest and
Git installation identity to distinguish it. The updated tag allows model-free `models: []`
descriptors but does not provide a production NLI judge.

## Frozen-Prediction Interval Calibration

[`calibration.py`](calibration.py) fits interval radii on earlier training-event residuals
and writes adjusted answers for later calibration events. This is an external development
workflow for frozen local predictions; it does not alter the submission defaults
or ship fitted outcome-derived parameters in the public image. Unverified model reports, test
splits, overlapping windows, incomplete runs and mismatched prediction settings are refused.

First run the unchanged grounded policy on separate train and calibration manifests with
the same code, toolkit and retrieval settings. Reports predating recorded prediction settings
must be regenerated. Then run:

```bash
python -m baselines.evaluation.calibration \
  --fit-manifest /private/evaluation/train-manifest.json \
  --fit-report /private/evaluation/train-run/report.json \
  --apply-manifest /private/evaluation/calibration-manifest.json \
  --apply-report /private/evaluation/calibration-run/report.json \
  --out /private/evaluation/new-calibrated-run
```

The finite-sample quantile uses the maximum absolute error within each event. Correlated
entity rows, target views and repeated seeds cannot inflate the event count. Bins match the
declared family, target, units, horizon and roster size; insufficient events or missing bins
produce an error instead of an invented finite interval. The statistical coverage statement
assumes exchangeable event residuals, meaning that past and future error distributions can be
treated alike. Chronological financial data does not establish that assumption.

The tool verifies original answer and input hashes and recomputes fitting residuals through
the shared scorer. It writes all adjusted answers before opening application outcomes, then
rescoring uses the same official formulas. Points, labels, ranks and citations stay unchanged;
interval-width diagnostics accompany the new report. Fitted radii, original answers and reports
remain outside every public worktree. Changing intervals also changes the submitted hypothesis:
historical citations alone do not establish future bounds, and production NLI must be measured
separately. Local coverage on the calibration split is not a held-out or leaderboard claim.

For model predictions, also supply `--fit-runtime` and `--apply-runtime` with external local
launcher records. [`runtime.py`](runtime.py) defines their validation: each completed record
binds the exact report and manifest hashes to a clean code revision, a local launch command,
the runtime executable, weight shards, launcher identity, host and generation settings. The
executable and weight files are rehashed. Only the loopback port may vary between launches;
changing the context, backend or any other recorded setting requires new matching runs.
The model reports must include complete per-entity diagnostics with no grounded fallbacks.
Old launcher records lacking binary/report hashes cannot be used and must be regenerated.

These records are attestations from a trusted local launcher that owns the server, not
cryptographic proof of what an arbitrary remote service executed. The launcher must record
the actual command and effective settings, seal report hashes after evaluation, and mark
completion only after the owned processes have exited successfully. Calibration never
reuses grounded residuals for model predictions. Its artifact retains the runtime record
hashes and common model identity; it remains a development experiment outside the image.

## Internal acceptance and sealed batches

The [championship plan](../../docs/CHAMPIONSHIP-PLAN.md) defines G1/G2/G3 and the
[acceptance policy](acceptance-policy.json) owns their numerical thresholds.
[`acceptance.py`](acceptance.py) checks persisted evidence using the existing scorer and
comparator; it reports PASS, FAIL or UNMEASURED without changing the incumbent.
Missing measurements never become passes, and declared report totals cannot replace per-run evidence.

| Entry point | Responsibility |
| --- | --- |
| [faults](faults.py) | Exercise the real image with synthetic HTTP faults; verify recovery artifacts |
| [batch](batch.py) | Freeze identities and event reservations, run each role once, recheck receipts and decide |
| [lifecycle](lifecycle.py) | Own the development round budget, incumbent, promotion and rollback journal |
| [search](search.py) | Compare registered development candidates and select one for fresh acceptance |
| [confirmations](confirmations.py) | Seal and audit production batches against the original G2 pair |
| [production](production.py) | Bind an explicit official-format judge specification and local cache; verify drift |

Consult each module's `--help` for arguments. Registries, version declarations, manifests and
outputs follow the [private-data rules](#private-development-data) above. These tools do not
certify organizer approval, source licensing, first availability, prior human exposure or event
independence. Those facts require separate provenance evidence.

### Shared batch rules

Registration freezes the policy, before/after source and toolkit identities, input/truth digests,
seeds and expected roster. The version-file shape is validated by `snapshot` in [batch.py](batch.py).
An optional manifest `domain` supports stratified comparisons; acceptance requires it, and it
never becomes an agent input. Both roles must retain the same toolkit and evaluation profile.
The registered runner uses offline grounded prediction: smoke is the default, while production
scoring additionally requires the explicit judge configuration described below.

Event groups and input digests are reserved across renamed cases and changed seeds. Development
candidates may reuse development reservations, but those reservations cannot become fresh
acceptance, and acceptance reservations cannot become development data. Starting a role consumes
it even after a crash. Complete runs, including participant failures, receive a digest-bound
receipt; incomplete runs cannot produce a partial verdict. `batch decide` rechecks both receipts
and writes an exclusive decision, refusing changes to frozen identities or evidence.

Recovery evidence can be frozen for both roles at registration. The [fault runner](faults.py)
uses the real analyze process with invented inputs and an isolated loopback HTTP service.
It compares server-observed requests with the client ledger and checks complete outputs,
citations, fallback behavior and resource observations. Each recovery report must match its
role's immutable image and runtime source; missing evidence remains unmeasured and drift fails.
An engineering exercise can be rerun after repair in a new directory because it consumes no real
acceptance events. Recovery does not measure forecasting quality or production faithfulness.

The registry is an evaluator-owned audit trail, not protection against an owner rewriting all
its files. Saved decisions from another evaluator revision may fail recomputation; retain the
original evidence rather than rewriting it to force a transition.

### Development selection and fresh acceptance

1. Initialize the lifecycle with a clean frozen bootstrap version. Open a round with a hypothesis
   and an external budget. `validate_budget` in [batch.py](batch.py) defines the shared run/time
   limits. The journal records every execution reservation before work; errors and crashes do
   not refund it. The subprocess limit includes the report/shutdown allowance in `execution_budget`.
2. To measure development candidates within that budget, supply the development manifest and
   seeds at `lifecycle start-round`. This freezes its calibration-split roster, input/truth hashes,
   resolution bound and policy candidate cap. Use a dedicated selection manifest: the existing
   split name does not authorize mixing it with fitting data. Omitting these arguments supports
   acceptance of an already selected pair, without certifying its earlier search budget.
3. Register each candidate with the development purpose, then run both roles through `batch run`.
   Every candidate retains the round incumbent. Each pair must fit the remaining budget; leave
   enough for eventual acceptance. Failed or abandoned candidates count toward the cap.
4. Run `search select`. It rechecks receipts and persisted results for every completed candidate;
   unfinished candidates require explicit abandonment with a reason. Completed evidence cannot
   be discarded. Successful runs, positive event-mean gain and the policy's stratum floor are
   required; largest gain wins, with batch identity breaking ties. No eligible candidate means
   KEEP_INCUMBENT. This screening does not establish G2 and closes further development work
   in that round.
5. Register a fresh test-only batch and `lifecycle attach` it before either role starts. It must
   retain the current incumbent and a different clean candidate. In a search round it must match
   the selected version and seeds, with cutoffs after the development resolution bound. Execution
   and later replay recheck the selection evidence. Only one candidate may enter acceptance per
   round, even after abandonment or with spare budget.
6. Run both roles, then `batch decide` and `lifecycle resolve`. Resolve recomputes the decision
   from original artifacts and verifies the versions remain unchanged. Only G1/G2 PASS updates
   the development incumbent and retains the old version for rollback; failure or missing evidence
   closes the batch and keeps the incumbent.

`lifecycle status` derives state from the append-only hash-linked journal. A pending batch must
be resolved or abandoned before another selection or rollback; abandonment preserves event
reservations and spent budget. `close-round` retains history and cannot discard pending acceptance.
`rollback` requires a reason and verifies the previous version, preserving its existing qualification.
A restored bootstrap remains unqualified.

The round controls registered offline grounded/smoke work only, not arbitrary shell commands,
training, paid services or historical experiments. Standalone diagnostic registries remain usable,
but their already-started runs cannot later become budgeted lifecycle acceptance.

### Production confirmations

After G2, register fresh production batches using the original selected candidate **and original
baseline**. The explicit judge specification and already-cached artifacts must be outside public
worktrees. [production.py](production.py) validates the official-format specification, binds its
bytes and cache digest, pins offline evaluator loading, and checks each report's judge identity.
It does not download weights, authorize paid inference or establish organizer approval.

Before either confirmation runs, use `confirmations` with its sealing budget option to bind the
selection and both confirmation registrations. Sealing rechecks G1/G2, the fixed versions, seeds,
policy, identical production judge and disjoint event/input identities. The combined budget uses
the same run/time schema as development but is separate from the development round. In a lifecycle
registry, the selection must be the current G2 incumbent with no pending development acceptance.
A third or replacement confirmation cannot be substituted, and partial sealing refuses execution.

Run each role through `batch run`, decide each batch, then use the confirmation audit. It rechecks
original reports, receipts, both roles' production faithfulness and every batch's G1/G2 separately;
a pooled average cannot hide a failed batch. Organizer runtime equivalence and artifact eligibility
remain UNMEASURED until their evidence contracts are implemented and satisfied. This audit does
not declare a production candidate, deploy, submit or mark the overall goal complete.

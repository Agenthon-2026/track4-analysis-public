# Strong RAG baseline - BM25 retrieval and evidence-grounded prediction

## Executive summary (read this first)

This baseline retrieves evidence for each entity from the frozen corpus and predicts the target
defined by the task. By default, `analyze` calls the organizer-provided `$MODEL_ENDPOINT` using
`$MODEL_NAME`; the HTTP client honors the harness proxy environment. Without an endpoint, or with
`--mock`, it uses a deterministic offline reasoner. Both paths use the same task schema,
embargo filtering, exact citation offsets, and output validation on every task. Local GGUF
inference is an explicit development option and is separate from official model serving.
Passing a local output or citation check does not establish production NLI admission or
predictive quality.

## Run

The harness supplies the model environment listed in
[`SUBMISSION_CLI.md`](../../SUBMISSION_CLI.md#container-environment-contract-restricted-mode-set-by-the-harness).
The CLI accepts the same leading verb as the submission container:

```bash
python -m baselines.strong_rag_baseline.cli analyze \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/answer.json

# Explicit offline fallback, even when an endpoint is configured.
python -m baselines.strong_rag_baseline.cli analyze \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/answer.json \
  --mock
```

The image recipe is [`baselines/Dockerfile`](../Dockerfile), with
[`analyze.py`](../analyze.py) consuming the harness verb. It contains runtime Python modules;
test fixtures and development model weights are excluded. From the repository root:

```bash
docker build --platform linux/amd64 -f baselines/Dockerfile -t t4-analyze:latest baselines
bash baselines/smoke_image.sh /tmp/t4-out
```

`smoke_image.sh` builds the image and runs its offline fallback under `--network=none`. It exits
nonzero when Docker or its daemon is unavailable. The Python CLI command above can still be run
separately, but a local process does not verify the image. The container smoke also does not test
the production endpoint, NLI admission, or predictive quality.

For a local faithfulness preview, install and cache the judge dependencies, then run:

```bash
python faithfulness/judge.py --answer /tmp/answer.json --unit units/t4-EXAMPLE-eps-beat
```

`--unit` names the full unit directory, including its task, card, manifest and corpus. Record the
judge version and environment with any result. The organizers have
[announced pending changes to NLI normalization and calibration](https://github.com/Agenthon-2026/track4-analysis-public/issues/1#issuecomment-5534948217);
an unpinned local preview cannot prove official admission. Public practice units have no resolved
outcomes, so their smoke results do not measure prediction quality.

## Developer-machine GGUF

`--local-llama` or `T4_LOCAL_LLAMA=1` enables an optional llama.cpp server on loopback for local
experiments. The submission defaults do not enable it. Setup lives in
[`baselines/models/README.md`](../models/README.md); neither the server nor GGUF weights are
included in the submission image. The offline reasoner handles an unavailable local server.

```bash
python -m baselines.strong_rag_baseline.cli analyze \
  --local-llama \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/answer.json
```

Runtime retrieval, timeout, retry, seed and token settings are defined in
[`config.py`](config.py); CLI choices are defined in [`cli.py`](cli.py). The official path reads
`MODEL_ENDPOINT` and `MODEL_NAME`. Development overrides should not replace the model identity
provided by the harness.

## Official Submission Categories

Use `api` when calling the house endpoint. The organizer's
[BYO starter-pack contract](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/starter-packs/track4/AGENTS.md#bringing-your-own-model-adapter-only-rank--64)
describes one LoRA adapter, rank at most 64, on the organizer-hosted Nemotron base. The organizer
extracts the adapter, starts the model server, and supplies the same endpoint contract; the
participant image does not start vLLM or ship full reader weights. This supersedes the older
full-weights description still present in `SUBMISSION_CLI.md`.

The [descriptor guide](https://github.com/Agenthon-2026/Agenthon2026-public/blob/main/starter-packs/track4/SUBMISSION-DESCRIPTOR.md)
also documents a model-free deterministic declaration using the legacy `byo-small` category
and `models: []` (C5 1.1.0, toolkit tag `v2.4.0`). Use it only when no model is used;
do not invent a placeholder model entry. This baseline's default endpoint path is an `api` entry.

## Design

| Module | Role |
|---|---|
| `indexer.py` | Reads corpus text and creates chunks with exact character offsets |
| `retriever.py` | BM25 retrieval over dated, embargo-eligible documents |
| `evidence.py` | Entity-bound excerpts, explicit series-column tables, evidence IDs and provenance ledger |
| `tables.py` | Strict dated columns, source-bound historical differences and explicit percent-to-bps conversion |
| `client.py` | HTTP model calls through the configured endpoint, plus a mock client for tests |
| `prompts.py` | Task-aware JSON schemas and requests for predictions with evidence IDs |
| `schema.py` | Reads target type, allowed labels, units and interval requirements |
| `quantities.py` | Validates finite numeric targets, units and declared domains |
| `reasoner.py` | Deterministic fallback that interprets evidence in the task's target context |
| `agent.py` | Coordinates model inference, evidence validation and fallback |
| `formatter.py` | Assembles and validates the complete entity roster |
| `validation.py` | Shared checks for predictions, citations, dates and entity coverage |
| `local_server.py` | Optional development-only llama.cpp launcher |

Retrieval is lexical; a dense encoder and learned calibration head are not shipped. Predictions
and intervals must refer to the requested target and its units. Finding a number in a passage
does not establish that it forecasts the requested quantity.
The model path uses the bounded evidence packet from `evidence.py`. It selects source-bound IDs;
the program restores the original text and offsets instead of asking the model to copy quotes.
Once candidates are entity-bound, metric queries avoid repeating the entity name; bare names
and recognized administrative headings are excluded without requiring short facts to match a verb list.
An explicit series-column table retains its header and dated rows; other columns are context,
not values attributed to the requested series. Dated metric tables can also bind through an
unambiguous document title. Complete rows are retained within the excerpt budget; summaries
state the selected date range, which can be shorter than the full source history.
Lexical numeric annotations remain unlinked mentions. Separately, `tables.py` supplies the prompt
and provenance ledger with arithmetic for exact series columns or matching target columns.
Missing or unsupported numbers and duplicate or reversed dates prevent a numeric summary.
Unknown units remain null; percent-to-bps conversion requires an explicit source declaration.
Historical extrema are descriptive context, not calibrated prediction intervals. Model forecasts
and interval bounds still require validation and independent outcome evaluation.
The HTTP client requests a strict JSON schema. Unsupported schemas, incomplete responses,
invalid predictions or unknown IDs trigger an explicit fallback, without unconstrained retries.
Custom clients may retain the original `complete` interface and exact-quote response format;
both formats undergo citation and prediction validation. These checks do not establish entailment.
The fallback cites verbatim observations; its calculations and uncalibrated interval
assumptions are recorded separately in `notes.fallback_rationale`.

## Verification

Run the tests under `tests/` for task generalization, endpoint routing, citation handling and
offline fallback. Run `../tests/test_submission_image.py` for the entrypoint and container-script
contracts, then use `smoke_image.sh` on a Docker host for an actual container run. Keep production
NLI admission and evaluation on independently resolved outcomes as separate measurements; neither
is claimed by these checks.


## Per-unit request accounting

The [client](client.py) shares one request budget across all entities in an analyze
run. It counts each attempt before I/O, including failed, truncated and retried
requests; after exhaustion, later entities use the grounded fallback without more
HTTP calls. [Config](config.py) owns the conservative request/output ceilings and
clamps environment output settings to the selected House limit. Those ceilings
are not a claim about the organiser's still-pending BYO or input accounting.

Optional diagnostics include a bounded request ledger: sequence, status, byte
count, timing and request digest. Provider-reported token counts are recorded when
valid and otherwise remain null. It contains no prompts, responses, endpoint URLs
or credentials. Offline runs record zero attempts; an uninstrumented model client
records no ledger rather than fabricating zero usage. This is client evidence,
not an audited proxy bill or a production-runtime equivalence result.

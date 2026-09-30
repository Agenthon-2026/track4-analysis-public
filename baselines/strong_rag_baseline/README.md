# Strong RAG baseline — BM25 retrieval + house model over `$MODEL_ENDPOINT`

## Executive summary (read this first)

The track's reference retrieval-augmented agent (Baseline 3 in `../README.md`): for each entity
it retrieves the top-K span-level chunks from the frozen corpus with BM25 (embargo enforced at
retrieval time), sends them with the entity's tabular features to the house model at
`$MODEL_ENDPOINT`, and turns the model's quoted evidence into claims with **exact
`(doc_id, span_start, span_end)` citations** — model-supplied offsets are never trusted; quotes
are located as verbatim substrings of the corpus, and anything ungroundable is dropped rather
than cited loosely. One agent for all units, no per-unit tuning; deterministic given the model
pin and seed (temperature 0, fixed seed, stable tie-breaks).

**Status: scaffold.** Fully runnable end-to-end with `--mock` or any local OpenAI-compatible
server; quality acceptance (beats `baseline_agent/`, ≥0.80 faithfulness under the pinned judge)
waits on the staging `$MODEL_ENDPOINT`.

`--mock` is a **wiring check, not a prediction**. It answers from the prompt it is handed:
it quotes a verbatim slice of the top retrieved excerpt, so the quote grounds to a real span
through the same path a real model's quote takes, and every row comes back with at least one
grounded claim. What it does not do is forecast — `point_forecast` is `0.0` except on ranking
units, where a constant vector would be the degenerate answer. Use it to prove
retrieval → prompt → parse → ground → assemble works end to end; do not read its numbers.

## Run

```bash
# standard interface contract
python -m baselines.strong_rag_baseline.cli \
  --task   units/t4-EXAMPLE-eps-beat/task.json \
  --corpus units/t4-EXAMPLE-eps-beat/corpus \
  --out    /tmp/answer.json

# wiring smoke run without any model server
python -m baselines.strong_rag_baseline.cli --task ... --corpus ... --out ... --mock
```

Environment:

| Var | Meaning | Default |
|---|---|---|
| `MODEL_ENDPOINT` | House route origin (harness-injected at scoring time); requests go to `$MODEL_ENDPOINT/v1/chat/completions`. A local URL already ending in `/v1` also works | — (required unless `--mock`) |
| `MODEL_NAME` | model id sent in the request — this is what the harness injects (see `SUBMISSION_CLI.md`, container environment contract) | empty |
| `MODEL_ID` | local-dev fallback for `MODEL_NAME`; read only when `MODEL_NAME` is unset | empty |
| `MODEL_TOKEN` | per-unit bearer credential (harness-injected at scoring time); sent as `Authorization: Bearer` on every request. Optional locally | none |
| `T4_SEED` | seed forwarded to the model | `20260731` |
| `T4_TOP_K` | retrieved chunks per entity | `10` |
| `T4_MODEL_TIMEOUT_S` / `T4_MODEL_RETRIES` | per-call timeout / retry count | `60` / `3` |

Local model example: `ollama serve` + `MODEL_ENDPOINT=http://localhost:11434/v1 MODEL_ID=qwen2.5:7b`.

## Design

| Module | Role |
|---|---|
| `indexer.py` | One chunk per corpus span; global offsets follow the scorer's join-with-space convention, so every chunk is citation-ready as-is |
| `retriever.py` | Pure-Python Okapi BM25; docs with missing or post-cutoff `doc_date` dropped before scoring; ties break by `(doc_id, span_start)` |
| `client.py` | stdlib HTTP client for `$MODEL_ENDPOINT/v1/chat/completions` with the `MODEL_TOKEN` bearer (temp 0, seed; retries network errors, timeouts, HTTP errors and a malformed response envelope, then raises `ModelCallError`; the first HTTP 401, 403 or 404, or an endpoint with no URL scheme, raises `ModelConfigError` at once, with no retry, and ends the run; the exception is a 403 whose JSON body has `error.code` `grant_denied` and `error.message` `request admission refused`, the House's answer once the unit's request allowance is used up (or for an expired or unknown grant): it raises `ModelBudgetExhausted`, no further request is sent, and that entity and every later one get fallback rows, listed in `notes.budget_denied_entities`, so the unit still finishes with an answer and exits 0, before or after a reply (a non-zero exit is charged to the team as a crashed container); `notes.budget_refused_before_any_reply` is true when no request in the unit got a reply before the refusal, and `notes.budget_refused_before_any_usable_reply` when none got a usable one (failed calls, unusable replies and their retries do count toward the 25 requests); a 403 with code `grant_denied` and the message `request model does not match credential` (a wrong model name), or any 403 whose body cannot be parsed, is a configuration error and ends the run like any other 401, 403 or 404) + `MockModelClient` for tests |
| `prompts.py` | Per-target-type prompt; demands one JSON object with verbatim quotes |
| `span_finder.py` | Locates quotes as exact substrings (length-preserving curly-quote normalization); never trusts model offsets |
| `agent.py` | Orchestration; ungroundable quotes fall back to the source chunk's known-good offsets or are dropped; off-vocabulary labels and missing intervals get deterministic fallbacks; a null `label` or `point_forecast` is left out, never written; a model reply that cannot be used (no JSON, bad JSON, wrong field types, a NaN, infinite or float-overflowing `point_forecast`, JSON nested too deep, no number on a regression or ranking unit) or a model call that still fails after the client's retries gives that one entity a fallback row — placeholder `point_forecast` 0.0, first allowed label, one claim quoting corpus text verbatim — logged to stderr and listed in `notes.fallback_entities`, so the other entities keep their rows; a row whose model evidence does not ground cites the top retrieved excerpt verbatim (with nothing retrieved, the start of the newest document dated on or before the cutoff, as `baseline_agent` chooses it; with no such document, no claim) and is listed in `notes.fallback_claim_entities`; a missing interval, or one with a NaN, infinite or too-large bound or with lo above hi, is replaced by a band of point ± max(\|point\|/2, 1) around the model's kept forecast, logged to stderr and listed in `notes.interval_fallback_entities`; a failed call's entity is also listed in `notes.call_failed_entities`; when every entity's model call fails (after the client's retries, not a configuration error), the unit still finishes with fallback rows and exits 0, with `notes.every_model_call_failed` true; when the model answered but no entity's reply can be used, the run ends with no answer (unless the request allowance ran out, see `client.py`) |
| `formatter.py` | Final answer assembly + hard self-check (spans resolve, intervals complete, `notes` is an object) |

**BM25 only, no dense retrieval** (deviation from the Baseline-3 sketch in `../README.md`): the
eval sandbox's restricted network cannot fetch embedding weights at run time, so a lexical index
keeps the agent reproducible everywhere. The binding constraint is build-time vendoring: nothing
can be downloaded at run time, and bundling embedding weights for a dense index is an additional
neural checkpoint under the [artifact policy](../../docs/ARTIFACT-POLICY.md), which needs
organizer approval. The chunking already
targets the corpus's natural citable units (rendered-table NOTES lines, per-span passages),
which recovers much of what dense retrieval would add on these corpora.

## Acceptance (tracked, not yet runnable)

- [ ] Schema PASS + embargo PASS on the public practice unit(s)
- [ ] ≥0.80 citation faithfulness under the pinned judge
- [ ] Predictive quality strictly above `baseline_agent/`
- [ ] Runs as-is on `sample-tasks/track4-analysis/` and passes `evaluation/check_submission.py`

All four wait on the staging `$MODEL_ENDPOINT` and the sample-tasks export.

# AGENTS.md — Track 4 Public Repo

## Executive summary (read this first)

This file tells an AI coding agent how to work safely in the Track 4 public repository. Track 4
is "Explainability" — agents predict a label or value per row in a table
of entities, grounded in a frozen evidence corpus. Three rules dominate everything else: **(1)**
never put answer keys, resolved outcomes, or adversarial variant details in this folder — they
belong only in the private repo; **(2)** every new document must open with a plain-English
executive summary (the project style guide, `DOC-STYLE.md`, lives in the organisers' hub repository
and is not public; the shipped documents here are the working examples of it); **(3)** scoring math
lives in the shared `qfbench2-common` package — call it, do not copy or rewrite it here.

**Track 4 is maintained by the Agenthon 2026 organisers.** Questions about this repository go to
the role address the shipped unit cards carry in `author_email`: `qfbench@neurips2026.org`.

---

## Hard rules for this repo

1. **Firewall.** This repository is the public half of Track 4. No file here may contain:
   - A resolved outcome (the true `label`, `point_forecast`, `true_eps`, `true_direction`, etc.)
   - An `outcome.json` file anywhere
   - Adversarial variant construction details or logs
   - `adversarial_log.jsonl`, `perturbation_log.jsonl`, or any file named `oracle_*`,
     `solution_*`, `answer_key_*`, `expected_*`, `reference/outcome*`

   **Two checks enforce this, and you need both.**
   `qfbench2_common.manifest.assert_public_safe` (run by the `validate-units` CI job, and by
   `qfbench2 manifest assert-public-safe <unit_dir>` locally) rejects `solution.py`, `solve.sh`,
   `*oracle*`, `answer_key*` and the oracle directories — but its answer-material rule
   (`reference/`, `adversarial_variants/`, `outcome*.json`, `expected*`) applies only to units that
   are **not** `public-dev`, and every published Track-4 unit is `public-dev`. Measured 2026-08-28
   by planting one file at a time in a real unit: it accepts `outcome.json`, `expected_*`,
   `reference/`, `adversarial_variants/`, `adversarial_log.jsonl`, `perturbation_log.jsonl` and
   `solution_leak.py`. Widening that is a hub decision.
   `baselines/tests/test_units_carry_no_answer_material.py` closes the gap for this repository,
   rejecting the whole list above anywhere under `units/`, at any depth. It is standard library
   only and runs in the secret-free `firewall` job, and it carries every accepted-by-the-toolkit
   plant as an executable control. Do not narrow it without narrowing this rule first.

2. **Inherit, do not copy.** Scoring functions live in the shared toolkit, whose repository path
   is `Agenthon2026-public/common/qfbench2_common/scoring/` — another repository, not this one. The track's
   `scoring/scoring.py` calls them via `from qfbench2_common.scoring import ...`. A new scoring
   function belongs in the toolkit — do not reimplement it here.

3. **Standard gate names.** The four admissibility gates are always named `g0_integrity`,
   `g1_schema`, `g2_cutoff_resource`, `g3_domain_semantics`. The submission verb is `analyze`.
   Do not invent new names.

3b. **`build_verifier` is the RANKABLE factory and it constructs a production judge or refuses.**
   It used to build no judge at all: `ctx.get("judge")` was `None` under the real driver, the
   faithfulness block was skipped, and `_score` read `ctx.get("_faithfulness", 1.0)` — a missing
   judge meant *perfect* faithfulness, and the gate the whole track exists for never ran. There is
   now exactly one non-rankable factory, `build_smoke_verifier`, it is separately named, it stamps
   `judge_mode="smoke"` and `rankable=False`, and **no environment variable reaches it**. Do not
   add a fallback from one to the other; a missing judge is an organizer failure with no score.
   The shared `qfbench2-smoke` runner picks between them by NAME through
   `qfbench2_common.smoke.resolve_verifier_factory` — `--profile smoke` (default) takes
   `build_smoke_verifier`, `--profile production` takes `build_verifier` and nothing else — so
   keep both names exported from `qfbench2_track_analysis.scoring`. Track 4 used to ship a
   module-level stopgap for this; it is deleted, because the shared runner does the selection now.

3c. **The entailment hypothesis comes from the SUBMITTED PREDICTION, never from claim text.**
   `hypothesis.py` derives it from the label/forecast/rank/interval plus the trusted task schema.
   Quoting the corpus back at itself used to score faithfulness 1.0 on a prediction that was wrong
   in both label and value.

4. **Writing standard.** Every new Markdown file must start with `## Executive summary (read
   this first)` followed by 3–8 plain-English sentences. Define jargon on first use or link to
   `docs/CONCEPTS.md`, which is this track's definitive glossary. The competition-wide GLOSSARY
   covering all four tracks is published with the shared toolkit:
   `Agenthon-2026/Agenthon2026-public`, file `docs/GLOSSARY.md`. The project style guide,
   `DOC-STYLE.md`, is organiser-internal and is not published with this repo.

   **When a fact lives in an artifact, prose links to it and does not restate it.** A schema
   field list, a card value, a CLI flag, an env-var name — name the artifact's path and let the
   reader (and CI) read the artifact: the answer shape is `templates/answer.example.json`
   validated against `analysis.schema.json`, the scoring parameters are
   `units/t4-EXAMPLE-eps-beat/card.toml [scoring.params]`, the container environment is
   SUBMISSION_CLI.md's
   container-environment table. Hand-restated facts are how the 2026-08 prose-drift class
   (39 findings) happened; the guards in `baselines/tests/test_docs_match_artifacts.py` fail CI
   on the highest-traffic violations (thresholds, repo paths in fenced blocks, env-var names,
   the `[environment]` grant), and they fail closed if a patrolled document goes missing.

5. **Corpus integrity.** Do not modify any file under `units/*/corpus/` or `units/*/manifest.json`
   without running the full schema + embargo check CI suite afterwards. The corpus is immutable
   once a task enters the test set.

6. **Target type generalization.** Track 4 supports three target types: `classification`,
   `regression`, and `ranking`. When editing `scoring/scoring.py`, ensure all three branches
   are handled. The target type is declared as `target.type` in `task.json` and read from
   `target_type` under `[scoring.params]` in `card.toml`. Do not assume classification only.

---

## What is safe to edit here

- `docs/` — the participant-facing explainers: `CONCEPTS.md` (terms), `CATEGORIES.md` (the
  published prediction families) and `AUTHORING-GUIDE.md` (a unit's on-disk layout, field by field)
- `baselines/` — the example agent code and Dockerfile
- `templates/` — annotated example `card.toml` / `task.json` / `manifest.json` / `answer.example.json`.
  These are ILLUSTRATIONS. The shipped exemplar unit is authoritative, and
  `baselines/tests/test_template_card_matches_the_exemplar.py` pins the template's scoring
  parameters, compute grant and agent budget to the exemplar's card so the two cannot diverge
- `faithfulness/serving/README.md` — public documentation of the served-judge contract. The
  organiser deployment (image, manifests, endpoint, token) is not published here and must not be
- `faithfulness/judge.py` — the NLI judge public interface (do not change model IDs without
  updating both the code and `docs/CONCEPTS.md`)
- `scoring/scoring.py` — a re-export shim; the implementation is `qfbench2_track_analysis/`
- `qfbench2_track_analysis/` — the ONE Track-4 scoring implementation (do not change the gate
  names, the composite formula, the metric domain `[-0.27, 1.0]` or the worst-case value
  `W = -0.27` without a review by the track lead)

---

## Quick checks before calling something done

```bash
# From the root of this repository:
python -m pytest scoring/ faithfulness/

# The shared CLI, with no Track-4 stopgap in front of it. `--profile smoke` is the DEFAULT and
# resolves `build_smoke_verifier` (non-rankable preview); `--profile production` asks for
# `build_verifier`, the rankable factory, which refuses without a pinned production judge. The
# chosen factory name is printed with the verdict, so "passes smoke" and "would rank" are
# distinguishable in the output.
qfbench2-smoke units/t4-EXAMPLE-eps-beat /tmp/out --track analysis

# assert_public_safe RETURNS its findings; it never raises. The `python -c` form this
# line used to carry exits 0 even on a planted solution.py -- measured 2026-08-24. Use
# the CLI, which checks the return value and exits 1.
qfbench2 manifest assert-public-safe <unit_dir>

# ...and the repo-local half of the firewall, which rejects the answer material the shared
# check exempts for a public-dev unit (hard rule 1). Standard library; no toolkit needed.
python -m pytest baselines/tests/test_units_carry_no_answer_material.py
```

All three must pass before any commit to the public branch.

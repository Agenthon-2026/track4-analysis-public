## Executive summary (read this first)

Track 4 permits external data for offline training within an otherwise permitted submission
category. All task-specific features and labels used for fitting, adaptation, model selection
or calibration must have been available by the relevant task cutoff. Evaluation inputs and
citations remain limited to the supplied task and official frozen corpus. A narrow exception
applies only to the general-purpose pretraining of explicitly approved Nemotron base revisions.
This clarification changes no scoring formula, judge, threshold, submission schema or category.

**Policy revision: 2026-09-10.1.** Read alongside the [submission interface](../SUBMISSION_CLI.md)
and [competition data rules](https://www.agenthon.net/rules/).

## External training and historical cutoffs

For example, an approved adapter may be trained offline using a public, licensed dataset whose
features and labels were available before every task cutoff where it will be used. A label
released after a task's cutoff is forbidden for that task even if it is used only to choose a
model or calibrate a prediction interval.

External training data must be public, lawfully obtained and properly licensed. Nonpublic
employer/sponsor data and embargoed data require express written authorization. All task-specific
features and labels, including material used for adaptation, selection and calibration, must
have been available by each relevant cutoff. A historical observation date is insufficient
when the value or revised release became available later.

At evaluation time, use only the supplied task and official frozen corpus as inputs and
evidence. Offline-training permission does not authorize importing external documents into
the inference corpus, citing them, fetching data, or storing task-answer lookups. It does not
expand the allowed model or artifact categories: the [adapter-only BYO contract](../SUBMISSION_CLI.md#adapter-only-byo)
continues to apply.

## Required provenance record

Keep an `ARTIFACT_PROVENANCE.md` record with the source used to build your image, available
for organizer verification. Record sources, dataset versions, licenses and first-availability
dates. Identify the data used separately for fitting/adaptation, model selection and interval
calibration, together with the immutable model/adapter revisions or checksums. Explain how
the selected artifact respects each task cutoff where it is used.

Use the existing model disclosure in `submission.json` and keep base and adapter training
histories clear in the provenance record. Do not invent descriptor fields or placeholder
model entries. The provenance record is reviewed documentation, not a new descriptor key;
the current descriptor validator does not establish whether training data were eligible.
Do not add a sidecar to the descriptor upload ZIP unless the upload instructions request it.

## Exception limited to the approved Nemotron base

Only the exact organizer-approved Nemotron base revisions identified in the model-access
release receive an exception for their general-purpose pretraining on historical tasks.
This permits those base weights; it does not permit task-specific post-cutoff fitting,
adaptation, model selection, calibration or additional data. Declaring another model's
training cutoff does not qualify it for this exception. Use the approved revision list
before treating any particular base as covered.

## Executive summary (read this first)

Track 4 permits the limited local numerical artifacts below alongside the approved House model. Language-model serving is the House model only (bring-your-own models and adapters are not part of this competition, ruling of 2026-09-18); that rule does not prohibit fitted non-neural prediction or calibration parameters. All fitting, selection and calibration data must meet the task's information cutoff and the existing training policy. Additional neural checkpoints require separate organizer approval. Evaluation inputs and citations remain limited to the supplied task and official frozen corpus. This clarification does not change scoring, resource grants or the descriptor schema.

## Permitted local artifacts

| Artifact | Permission and conditions |
|---|---|
| Ordinary statistical code, deterministic preprocessing and numerical transformations | Permitted; pin package/source versions. A package name does not authorize every model it can load. |
| Fitted linear models, decision trees, XGBoost/LightGBM/CatBoost and other non-neural predictors | Permitted with disclosure of the actual learned models and their fitting/selection data. No post-cutoff labels or stored task-answer lookup. |
| Calibration parameters, thresholds, covariance estimates and prediction-interval calibration | Permitted under the same cutoff for fitting, selection and calibration. A pre-cutoff training set does not authorize later calibration labels. |
| Tokenizer-only vocabularies/configuration, static dictionaries and numerical lookup tables | Permitted when otherwise eligible under the data rules, without additional neural weights or stored task answers. |
| BM25 or other non-neural indexes used for evidence retrieval | May index the supplied task's official frozen corpus. This does not authorize importing external documents or citing an external inference corpus. |
| Neural forecasting/classification models, neural embeddings, neural rerankers or additional language models | Require separate express approval; being auxiliary or non-LLM is not automatic eligibility. |

An unchanged House model combined with these permitted local artifacts uses the API execution mode. Describe every learned local model with the existing `models[]` entry and `access: "local"`; include the House disclosure when used. Pure code and static assets belong in provenance documentation rather than fictitious model entries. This policy does not create a new model-free Track 4 category.

For an approved LoRA submission, the existing one-adapter/base/rank contract still applies, and the serving route must separately be available. Full language-model weights, a second language model or adapter and a participant-run language-model server are not authorized by this clarification.

## Provenance and task cutoffs

Follow `docs/TRAINING-POLICY.md`. Keep `ARTIFACT_PROVENANCE.md` with the source used to build the image, available for organizer verification. Record each learned artifact's immutable revision or checksum, license, sources and first-availability dates, distinguishing fitting, selection and calibration data. State how it respects each relevant cutoff.

At evaluation time, use only the supplied task and official frozen corpus for inputs and citations. Do not fetch data or import external evidence documents. The narrow general-purpose-pretraining exception applies only to the approved House base revision; it does not excuse participant adaptation, selection, calibration or another model's data history.

Use the existing descriptor fields without adding a license/provenance field to a model row or an unsolicited file to the upload ZIP. Format validation does not certify the truth of provenance. All permitted artifacts share the same actual compute and storage limits; no GPU, disk, memory or network allocation is added here.

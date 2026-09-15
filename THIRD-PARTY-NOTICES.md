# Third-Party Notices

This project incorporates or builds on the components below. **Each retains its own license**, and
that license governs that component — it is not superseded by this repository's `LICENSE` or
`DATA-LICENSE.md` (MIT). This file is a convenience summary; the authoritative license for any
released data file is the `license` field in its `manifest.json`.

## Software

- **ABIDES** (organizer-maintained fork) — **BSD-3-Clause**. Used by Track 3. The original ABIDES
  code remains under BSD-3-Clause and may be used commercially from upstream; **only the organizers'
  additions** (scenarios, regression harness, stylized-fact checker, throughput timer) are licensed
  under this repository's `LICENSE` (MIT).
- **Python dependencies** of `qfbench2-common` (NumPy, SciPy, jsonschema, pandas, PyArrow,
  transformers, PyTorch, and others) — each under its own license (BSD / Apache-2.0 / MIT / PSF).
  These are *installed*, not redistributed in this repository.

## Model weights

Track 4's citation-faithfulness judge is an ensemble of two DeBERTa-v3 NLI models, pinned in
`faithfulness/judge.py` (`NLI_MODEL_IDS`). **The two carry different licenses, and neither weights
license is the license of the data the model was trained on.** No weights are committed to this
repository; `faithfulness/judge.py` states they are pre-staged into the evaluation image at
`/model-cache/`, and redistributing them that way is what these terms govern. Licenses below are
the model cards' own `license` field, read 2026-09-01.

- **`cross-encoder/nli-deberta-v3-large`** (judge ensemble member 1 of 2) — weights
  **Apache-2.0**. Trained on SNLI (**CC BY-SA 4.0**) and MultiNLI.
  Source: https://huggingface.co/cross-encoder/nli-deberta-v3-large
- **`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`** (judge ensemble member 2 of 2) —
  weights **MIT**. Trained on MultiNLI, FEVER-NLI, ANLI (**CC BY-NC 4.0**, non-commercial),
  Ling-NLI and WANLI.
  Source: https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

## Datasets & corpora

- **QF-Bench** (Track 1 practice pool) — **CC BY-NC 4.0**. Redistributed here under those terms with
  attribution. Source: https://github.com (QFBench project) / the QFBench site.
- **SEC EDGAR** (incl. EDGAR-CORPUS / EDGAR-CRAWLER), **U.S. Federal Reserve** (FRED / ALFRED,
  H.10), **U.S. BEA**, **U.S. Treasury** — works of the U.S. Government, **public domain** (no
  copyright). The organizers' *curation, selection, labels, and questions* built from them are
  CC BY-NC 4.0.
- **JKP Global Factor Data** (jkpfactors.com) — per the providers' terms (academic use).
- **Global Macro Database** (globalmacrodata.com) — per the providers' terms.
- **Open Source Bond Asset Pricing / OSBAP** (openbondassetpricing.com; Dickerson, Mueller &
  Robotti; Dickerson, Nozawa & Robotti) — per the site's terms; **pin a specific release**.
  Only openly-posted factor / ML-panel files are used; **WRDS-gated panels are not redistributed**.

## Vendor data — referenced, NOT redistributed

- **Databento** (Track 3 limit-order-book data) and **Bloomberg** — commercial vendor data under
  their respective agreements. **Not committed** to these repositories; referenced by checksum and
  access instructions only. Any derived statistics are used strictly per the vendor's terms.

---

If you believe a component is mis-attributed or a license has changed, contact the Agenthon 2026
organizers before redistributing.

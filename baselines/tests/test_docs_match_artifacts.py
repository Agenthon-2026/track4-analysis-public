"""Prose-vs-artifact drift guards (prevention PR from the 2026-08-27 sweep).

The sweep found 39 cases of prose hand-duplicating an authoritative artifact (schema, card,
code) and then drifting from it. These tests pin the four highest-traffic fact classes to
their artifacts so the next drift fails CI at authoring time instead of surviving to a
participant. Standard library only, so the whole file runs in the secret-free ``firewall``
CI job (which runs ``pytest baselines``).

Each guard says which past finding it would have caught, and carries the *exemplars* of that
finding class as executable controls: a guard that cannot demonstrate it catches its own class
is worse than no guard at all, because it gets cited as proof.

When one of these fails, the fix is almost never "edit the test": either the prose or the
artifact changed unilaterally — make them agree, or link the prose to the artifact instead of
restating it (the standing editorial rule in AGENTS.md).

Fail-closed rule (repo rule R3): every patrolled document must be present. If a file named in
:data:`DOC_FILES` is deleted or renamed, these guards FAIL — they do not quietly patrol a
smaller corpus and report green.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Prose files whose factual claims these guards patrol. Every entry must exist (see
#: :func:`test_every_patrolled_document_exists`) — a missing entry is a guard that stopped
#: guarding, not a document that stopped mattering.
DOC_FILES = [
    "README.md",
    "SUBMISSION_CLI.md",
    "AGENTS.md",
    "docs/CONCEPTS.md",
    "docs/CATEGORIES.md",
    "docs/AUTHORING-GUIDE.md",
    "faithfulness/serving/README.md",
    "baselines/README.md",
    "baselines/strong_rag_baseline/README.md",
    "baselines/guardrails_example/README.md",
]

#: Non-prose artifacts the guards read. Same fail-closed rule.
#:
#: The card guard reads the EXEMPLAR UNIT's card, not an authoring template. The template was an
#: organizer artifact that participants never received, so prose could agree with it and still
#: disagree with every card actually shipped. `units/t4-EXAMPLE-eps-beat/card.toml` is the card a
#: participant holds, and it carries the same [scoring.params], [agent] and [environment] blocks.
ARTIFACT_FILES = [
    "units/t4-EXAMPLE-eps-beat/card.toml",
    "SUBMISSION_CLI.md",
]

#: The baseline READMEs guard 3 patrols. Same fail-closed rule.
BASELINE_READMES = [
    "baselines/README.md",
    "baselines/strong_rag_baseline/README.md",
    "baselines/guardrails_example/README.md",
]


def _require(paths: list[str], what: str) -> None:
    missing = [rel for rel in paths if not (REPO / rel).exists()]
    assert not missing, (
        f"{what} named in this guard no longer exist(s): {missing}. A guard whose subject has "
        "vanished must fail, not pass — either restore the file or delete it from the list in "
        "a commit that says why (repo rule R3)."
    )


def _card() -> dict:
    _require(["units/t4-EXAMPLE-eps-beat/card.toml"], "artifact file(s)")
    with open(REPO / "units" / "t4-EXAMPLE-eps-beat" / "card.toml", "rb") as fh:
        return tomllib.load(fh)


def _doc_texts() -> list[tuple[str, str]]:
    _require(DOC_FILES, "patrolled document(s)")
    return [(rel, (REPO / rel).read_text(encoding="utf-8")) for rel in DOC_FILES]


# --------------------------------------------------------------------------- #
# 0. The guards fail closed                                                    #
# --------------------------------------------------------------------------- #
# The original version of this file skipped any DOC_FILES entry that did not exist, so
# `mv README.md /tmp` turned four guards green while a real violation sat in the tree. That is
# a decoration, not a guard. Presence is now its own named check as well as a precondition
# inside every helper.

def test_every_patrolled_document_exists() -> None:
    _require(DOC_FILES, "patrolled document(s)")
    _require(ARTIFACT_FILES, "artifact file(s)")
    _require(BASELINE_READMES, "baseline README(s)")


# --------------------------------------------------------------------------- #
# 1. Scoring thresholds quoted in prose match the card — PER PARAMETER         #
# --------------------------------------------------------------------------- #
# The first version pooled every card value into one allowlist, so prose could swap two
# parameters' values and stay green: faithfulness_threshold=0.5, interval_level=0.70,
# tau_citation=0.3 and an inverted composite_weights=[0.3, 0.7] all passed. Each parameter now
# gets its own value set, and a number counts as "quoted" only when it is *bound* to the
# parameter by adjacency (assignment, colon, or the "0.5 (`tau_citation`)" spelling), so an
# unrelated number elsewhere on the line neither excuses nor triggers a failure.

#: A gap between a parameter name and its value may contain punctuation and whitespace only.
#: It can never contain a digit, so a binding always takes the *nearest* number.
_GAP_AFTER = r"[^A-Za-z0-9]{0,12}"
#: A value may also precede its name, but only in the parenthetical spelling
#: ``0.5 (`tau_citation`)`` — a bare comma or semicolon between two different
#: parameters is not a binding (that mis-bound `w_cal = 0.30`; `interval_level`).
_GAP_BEFORE = r"[^A-Za-z0-9\n]{0,4}[(\[][^A-Za-z0-9\n]{0,4}"
_NUM = r"\d+(?:\.\d+)?"


def _spellings(value: float) -> set[str]:
    """The spellings a doc may legitimately use for one card value.

    0.80 -> {"0.8", "0.80", "80", "80.0"} (the percentage spelling is idiomatic in prose);
    600.0 -> {"600", "600.00"}.
    """
    out = {f"{value:g}", f"{value:.2f}", str(value)}
    if 0.0 < value < 1.0:
        pct = value * 100
        out |= {f"{pct:g}", f"{pct:.1f}"}
    if float(value).is_integer():
        out.add(str(int(value)))
    return out


def _bindings(text: str, name: str) -> list[tuple[int, str]]:
    """(character offset, numeric token) pairs where the token is bound to ``name``."""
    found: list[tuple[int, str]] = []
    for m in re.finditer(re.escape(name), text):
        after = re.match(rf"({_GAP_AFTER})({_NUM})", text[m.end():m.end() + 40])
        if after:
            gap = after.group(1)
            # In a FORMULA, a parameter name before `=` binds the formula's RESULT, not the
            # parameter. `W = -w_cal x interval_level = -0.27` was read as
            # "interval_level = 0.27" and red-lighted prose that is correct (0.3 x 0.9 = 0.27,
            # matching #28's DOMAIN_MIN). If an `=` already appears between the start of the
            # line and this occurrence, the name is on a right-hand side and what follows is a
            # result.
            line_start = text.rfind("\n", 0, m.start()) + 1
            on_rhs_of_a_formula = "=" in text[line_start:m.start()]
            # A soft-wrapped continuation ("interval_level|\n= 0.90") is a real binding; an
            # adjacent line that merely starts with a number is not.
            wrapped_ok = "\n" not in gap or "=" in gap or ":" in gap
            if wrapped_ok and not on_rhs_of_a_formula:
                found.append((m.start(), after.group(2)))
        before = re.search(rf"({_NUM})(?:{_GAP_BEFORE})\Z", text[max(0, m.start() - 30):m.start()])
        if before:
            found.append((m.start(), before.group(1)))
    return found


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _threshold_offenders(docs: list[tuple[str, str]], allowed: dict[str, set[str]]) -> list[str]:
    offenders = []
    for rel, text in docs:
        for name, values in allowed.items():
            for offset, tok in _bindings(text, name):
                if tok not in values:
                    offenders.append(
                        f"{rel}:{_line_of(text, offset)}: {name} quoted as {tok} "
                        f"(card allows {sorted(values)})"
                    )
    return offenders


def test_scoring_thresholds_quoted_in_prose_match_the_card() -> None:
    params = _card()["scoring"]["params"]
    timeout = _card()["agent"]["timeout_sec"]
    # The guard's baseline must match the artifact, or it guards nothing.
    assert params["tau_citation"] == 0.5
    assert params["faithfulness_threshold"] == 0.80
    assert params["interval_level"] == 0.90
    assert params["composite_weights"] == [0.7, 0.3]
    assert timeout == 600.0

    allowed = _allowed_threshold_spellings()
    offenders = _threshold_offenders(_doc_texts(), allowed)
    assert not offenders, (
        "prose quotes a scoring parameter with a value the card does not carry:\n"
        + "\n".join(offenders)
    )


def _allowed_threshold_spellings() -> dict[str, set[str]]:
    params = _card()["scoring"]["params"]
    weights = params["composite_weights"]
    allowed = {
        "tau_citation": _spellings(params["tau_citation"]),
        "faithfulness_threshold": _spellings(params["faithfulness_threshold"]),
        "interval_level": _spellings(params["interval_level"]),
        # README spells the two composite weights as w_acc / w_cal; each is pinned to its own
        # position, so an inverted pair fails here as well as in the list check below.
        "w_acc": _spellings(weights[0]),
        "w_cal": _spellings(weights[1]),
        # 600 s is also idiomatically written as "10 minutes"; both spellings are the card's.
        "timeout_sec": _spellings(_card()["agent"]["timeout_sec"]) | {"10"},
    }
    # composite_weights itself is order-sensitive and is checked as a list, not a scalar; a
    # bare mention with a single trailing number still has to be one of the two weights.
    allowed["composite_weights"] = _spellings(weights[0]) | _spellings(weights[1])
    return allowed


def test_composite_weights_quoted_in_prose_keep_the_cards_order() -> None:
    """`composite_weights = [0.3, 0.7]` uses only card values but means the opposite thing."""
    weights = [float(w) for w in _card()["scoring"]["params"]["composite_weights"]]
    offenders = []
    for rel, text in _doc_texts():
        for m in re.finditer(r"composite_weights[^A-Za-z0-9\[]{0,12}\[([^\]]*)\]", text):
            quoted = [float(t) for t in re.findall(_NUM, m.group(1))]
            if quoted != weights:
                offenders.append(
                    f"{rel}:{_line_of(text, m.start())}: composite_weights quoted as {quoted} "
                    f"(card says {weights}; order is [predictive_quality, interval_coverage])"
                )
    assert not offenders, "prose reorders or rewrites composite_weights:\n" + "\n".join(offenders)


# --------------------------------------------------------------------------- #
# 2. Repo-relative paths in fenced code blocks exist                          #
# --------------------------------------------------------------------------- #
# Kills the M8 class permanently: every one of those findings was a copy-pasteable command or
# path that failed for whoever copied it. The first version caught neither exemplar — `cd
# public` is a bare directory (no slash, so the path regex never saw it) and
# `tracks/track4-analysis/public/units/...` began with two directories that were missing from
# _TOP_DIRS. Both are now controls in test_guard2_catches_its_own_exemplars.

#: Directory names that begin a repo-relative path. `tracks` and `public` do NOT exist in this
#: repo — they are the monorepo-relative prefixes the M8 findings used, and they are listed
#: precisely so that quoting one fails.
_TOP_DIRS = (
    r"(?:tracks|public|baselines|docs|templates|units|scoring|faithfulness"
    r"|qfbench2_track_analysis|\.github)"
)
#: The boundary class includes `/` so a bad prefix cannot hide a good suffix: in
#: `tracks/track4-analysis/public/units/x` the leftmost match starts at `tracks`, and the
#: whole (nonexistent) path is what gets checked.
_PATH_RE = re.compile(rf"(?:^|[\s('\"=:,/])({_TOP_DIRS}/[A-Za-z0-9_.\-/]+)")

#: `cd <dir>` inside a fenced block must name a directory that exists here.
_CD_RE = re.compile(r"(?m)^\s*cd\s+([^\s;&|#]+)")


def _fenced_blocks(text: str) -> list[str]:
    parts = text.split("```")
    return parts[1::2]  # odd segments are inside fences


def _is_placeholder(tok: str) -> bool:
    return "REPLACE" in tok or "{" in tok or "<" in tok or "$" in tok or "…" in tok


def _fenced_path_offenders(docs: list[tuple[str, str]]) -> list[str]:
    offenders = []
    for rel, text in docs:
        for block in _fenced_blocks(text):
            for match in _PATH_RE.finditer(block):
                tok = match.group(1).rstrip(".,;:)'\"")
                if _is_placeholder(tok):
                    continue
                if not (REPO / tok).exists():
                    offenders.append(f"{rel}: {tok}")
            for match in _CD_RE.finditer(block):
                tok = match.group(1).strip("'\"").rstrip("/")
                if _is_placeholder(tok) or tok.startswith(("/", "~", "-", "..")):
                    continue  # absolute path, home, a flag, or a walk out of the repo
                if not (REPO / tok).is_dir():
                    offenders.append(f"{rel}: cd {tok}")
    return offenders


def test_paths_in_fenced_code_blocks_exist() -> None:
    offenders = _fenced_path_offenders(_doc_texts())
    assert not offenders, (
        "fenced code blocks reference repo paths that do not exist "
        "(a reader will copy these and have them fail):\n" + "\n".join(sorted(set(offenders)))
    )


def test_guard2_catches_its_own_exemplars() -> None:
    """The two M8 exemplars, as controls. Both passed the first version of this guard."""
    exemplars = {
        "cd public": "```bash\ncd public\npip install -r baselines/requirements.txt\n```",
        "monorepo-prefixed unit path": (
            "```python\n"
            "t = json.load(open('tracks/track4-analysis/public/units/"
            "t4-EXAMPLE-eps-beat/task.json'))\n```"
        ),
    }
    for label, doc in exemplars.items():
        assert _fenced_path_offenders([("<exemplar>", doc)]), (
            f"guard 2 does not catch its own exemplar ({label}); it would be cited as proof "
            "of a class it cannot detect"
        )
    # ...and the same commands written correctly for this repo must stay green.
    clean = "```bash\npip install -r baselines/requirements.txt\ncat units/t4-EXAMPLE-eps-beat/task.json\n```"
    assert not _fenced_path_offenders([("<clean>", clean)])


# --------------------------------------------------------------------------- #
# 3. Env vars in the baseline READMEs are part of the published contract      #
# --------------------------------------------------------------------------- #
# Would have caught H6 at authoring time: the reference agent documented (and read) MODEL_ID
# while the harness injects MODEL_NAME. Any env var either comes from SUBMISSION_CLI.md's
# container-environment table or is an explicitly documented local-dev knob below.
#
# The first version skipped the entire `T4_` namespace ("documented local knob namespace"),
# which is the one namespace this repo actually owns, and its name regex required an
# underscore, so single-word variables (PYTHONPATH, HOME, MODELNAME) were invisible. Both are
# fixed; the T4_ knobs are now individually allowlisted and each must be read by shipped code.

#: Local-dev / harness knobs that are deliberately NOT in the container contract. Add here
#: only with a justification comment — an unexplained addition is exactly the H6 pattern.
#: Every ``T4_*`` entry must be read by shipped code (see the test below).
_LOCAL_ENV_ALLOWLIST = {
    "MODEL_ID",  # strong-RAG local-dev fallback for MODEL_NAME (H6 fix)
    "MODEL_TOKEN",  # strong-RAG local-dev bearer token; not harness-injected
    "QFBENCH_GPU_DEVICE",  # worker-side GPU pin, documented in the GPU caveats
    "TRANSFORMERS_CACHE",  # judge model-cache override, local runs
    "PYTHONPATH",  # smoke-run requirement documented in README
    "T4_JUDGE_BACKEND", "T4_JUDGE_URL", "T4_JUDGE_TOKEN",  # served-judge knobs (faithfulness/judge.py)
    "T4_SEED", "T4_TOP_K",  # strong-RAG determinism/retrieval knobs (strong_rag_baseline/config.py)
    "T4_MODEL_TIMEOUT_S", "T4_MODEL_RETRIES",  # strong-RAG per-call budget (config.py)
    "T4_TEMPERATURE",  # strong-RAG sampling temperature (config.py)
    "T4_UNIT_DIR",  # guardrails rail's unit selector (guardrails_example/rails/actions.py)
}

#: Matches `VAR`, `$VAR`, `${VAR}` and bare `$VAR` — including single-word names.
_ENV_RE = re.compile(r"`\$?\{?([A-Z][A-Z0-9_]{2,})\}?`|\$\{?([A-Z][A-Z0-9_]{2,})\}?")


def _contract_env_vars() -> set[str]:
    _require(["SUBMISSION_CLI.md"], "artifact file(s)")
    text = (REPO / "SUBMISSION_CLI.md").read_text(encoding="utf-8")
    m = re.search(r"Container environment contract.*?(?=\n###)", text, re.S)
    assert m, "SUBMISSION_CLI.md no longer has a 'Container environment contract' section"
    return set(re.findall(r"`([A-Z][A-Z0-9_]*)`", m.group(0)))


def _named_env_vars(text: str) -> set[str]:
    return {a or b for a, b in _ENV_RE.findall(text)}


def test_env_vars_in_baseline_readmes_are_in_the_contract() -> None:
    _require(BASELINE_READMES, "baseline README(s)")
    contract = _contract_env_vars()
    assert "MODEL_ENDPOINT" in contract and "MODEL_NAME" in contract  # sanity on the parse
    offenders = []
    for rel in BASELINE_READMES:
        text = (REPO / rel).read_text(encoding="utf-8")
        for var in sorted(_named_env_vars(text)):
            if var not in contract and var not in _LOCAL_ENV_ALLOWLIST:
                offenders.append(f"{rel}: `{var}`")
    assert not offenders, (
        "baseline READMEs name env vars that are neither in SUBMISSION_CLI.md's container "
        "contract nor in the documented local-dev allowlist (the H6 pattern):\n"
        + "\n".join(sorted(offenders))
    )


def test_local_env_allowlist_entries_are_read_by_shipped_code() -> None:
    """An allowlist nobody checks is how H6 survived. Every T4_ knob must exist in code."""
    sources = "\n".join(
        p.read_text(encoding="utf-8")
        for d in ("baselines", "faithfulness", "qfbench2_track_analysis", "scoring")
        for p in sorted((REPO / d).rglob("*.py"))
        if "tests" not in p.parts
    )
    orphans = [v for v in sorted(_LOCAL_ENV_ALLOWLIST) if v.startswith("T4_") and v not in sources]
    assert not orphans, (
        "these T4_ knobs are allowlisted for the baseline READMEs but no shipped module reads "
        f"them, so the docs promise a knob that does nothing: {orphans}"
    )


# --------------------------------------------------------------------------- #
# 4. [environment] values quoted in prose match the template card             #
# --------------------------------------------------------------------------- #
# Catches H7 recurrence: prose restating the compute grant with numbers the template card does
# not carry. The first version matched only the literal `cpus=`/`memory=`/`gpu=` spelling, so
# it patrolled a single templated line in the whole corpus and the drift that actually shows up
# in prose ("99 vCPUs", "128 GB of RAM", "no GPU", "CPU-only") sailed past. It now reads the
# prose spellings too, and checks `network` as well.
#
# (Template-vs-held-out-card alignment is decision D1's territory and cannot be tested from the
# public repo; this guard pins prose to the template so the two can only move together.)
#
# KNOWN GAP, deliberately not closed here: README.md's runtime-constraints row and
# baselines/README.md's wall-clock bullet both read "10 minutes per unit (CPU)" while the
# template card says `gpu = true`. A bare parenthetical "(CPU)" is not matched below, because
# making it fail would force this PR to rewrite a participant-facing compute promise, and which
# way it should be rewritten is decision D1/M6 (private issue #48), not a mechanical fix. The
# unambiguous spellings ("no GPU", "CPU-only", "GPUs are not provided", `gpu = false`) DO fail.

_CPU_RES = [
    re.compile(r"cpus?\s*[=:]\s*\"?(\d+)", re.I),
    re.compile(r"(\d+)\s*(?:v?CPUs?\b|CPU cores?\b|vcores?\b)", re.I),
    re.compile(r"(\d+)\s*(?:CPU\s+)?cores?\b", re.I),
]
_MEM_RES = [
    re.compile(r"memory\s*[=:]\s*\"?(\d+)\s*G", re.I),
    re.compile(r"(\d+)\s*(?:GB|GiB|G)\b[^.\n]{0,12}?\b(?:RAM|memory)\b", re.I),
    re.compile(r"\b(?:RAM|memory)\b[^.\n]{0,12}?(\d+)\s*(?:GB|GiB|G)\b", re.I),
]
#: Prose that asserts the grant has no GPU (True) or has one (False -> expects gpu = true).
_GPU_ABSENT_RES = [
    re.compile(r"gpu\s*[=:]\s*false", re.I),
    re.compile(r"\bno\s+GPUs?\b", re.I),
    re.compile(r"\bwithout\s+(?:a\s+)?GPUs?\b", re.I),
    re.compile(r"\bCPU[-\s]only\b", re.I),
    re.compile(r"\bGPU[-\s]?less\b", re.I),
    re.compile(r"\bGPUs?\s+(?:is|are)\s+not\s+(?:available|provided|allocated)\b", re.I),
]
_GPU_PRESENT_RES = [
    re.compile(r"gpu\s*[=:]\s*true", re.I),
    re.compile(r"\bGPUs?\s+(?:is|are)\s+(?:available|provided|allocated)\b", re.I),
]
#: Only the card's own quoted spelling. Bare ``--network=none`` is the docker flag for a
#: local smoke run, not a claim about the scoring grant.
_NETWORK_RE = re.compile(r"network\s*=\s*\"([a-z]+)\"", re.I)


def _environment_offenders(docs: list[tuple[str, str]], env: dict) -> list[str]:
    offenders = []
    for rel, text in docs:
        for i, line in enumerate(text.splitlines(), 1):
            for rx in _CPU_RES:
                for m in rx.finditer(line):
                    if int(m.group(1)) != env["cpus"]:
                        offenders.append(
                            f"{rel}:{i}: {m.group(0).strip()!r} (card says cpus = {env['cpus']})")
            for rx in _MEM_RES:
                for m in rx.finditer(line):
                    if f"{m.group(1)}G" != str(env["memory"]).upper().replace("IB", ""):
                        offenders.append(
                            f"{rel}:{i}: {m.group(0).strip()!r} (card says memory = "
                            f"{env['memory']!r})")
            for rx in _GPU_ABSENT_RES:
                for m in rx.finditer(line):
                    if env["gpu"]:
                        offenders.append(
                            f"{rel}:{i}: {m.group(0).strip()!r} denies a GPU (card says "
                            f"gpu = {str(env['gpu']).lower()})")
            for rx in _GPU_PRESENT_RES:
                for m in rx.finditer(line):
                    if not env["gpu"]:
                        offenders.append(
                            f"{rel}:{i}: {m.group(0).strip()!r} promises a GPU (card says "
                            f"gpu = {str(env['gpu']).lower()})")
            for m in _NETWORK_RE.finditer(line):
                if m.group(1).lower() != str(env["network"]).lower():
                    offenders.append(
                        f"{rel}:{i}: network = {m.group(1)!r} (card says {env['network']!r})")
    return offenders


def test_environment_grant_quoted_in_prose_matches_the_exemplar_card() -> None:
    env = _card()["environment"]
    offenders = _environment_offenders(_doc_texts(), env)
    assert not offenders, (
        "prose quotes an [environment] grant the exemplar card does not make (the H7 "
        "pattern):\n" + "\n".join(offenders)
    )


def test_guard4_catches_prose_spellings_not_just_key_equals_value() -> None:
    """The drift class guard 4 is named for, written the way docs actually write it."""
    env = _card()["environment"]
    drifts = [
        "The container gets 99 vCPUs.",
        "Each unit runs with 7 CPU cores.",
        "You get 512 GB of RAM.",
        "The grant is 512G memory.",
        "Agents run with no GPU.",
        "The card contract is CPU-only.",
        "cpus = 99",
        'memory = "512G"',
        'network = "open"',
    ]
    for drift in drifts:
        assert _environment_offenders([("<drift>", drift)], env), (
            f"guard 4 misses the prose spelling {drift!r} — it patrols a spelling nobody writes"
        )
    faithful = (
        f"The container gets {env['cpus']} vCPUs and {env['memory']}B of RAM; "
        f"network = \"{env['network']}\"."
    )
    assert not _environment_offenders([("<faithful>", faithful)], env)


# --------------------------------------------------------------------------- #
# 5. No participant-facing document offers a bring-your-own model path        #
# --------------------------------------------------------------------------- #
# Would have caught the 2026-09-18 residue at authoring time. The ruling of 2026-09-18 withdrew
# bring-your-own entirely, and the withdrawal landed in README.md's category section, in
# SUBMISSION_CLI.md and in docs/ARTIFACT-POLICY.md's summary -- while three documents went on
# offering the adapter contract in the same tree, ARTIFACT-POLICY.md contradicting itself between
# its line 3 and its line 18. That document is the one a participant was pointed at when they
# asked this exact question. A reader who found the offer first read a rule that no longer existed.
#
# The rule is about OFFERS, not mentions, so the guard reads SENTENCES: a sentence that raises
# bring-your-own, BYO or a LoRA/adapter submission must also carry the withdrawal. An earlier
# version of this guard listed offer phrasings instead, and a review showed it green on
# "BYO model submissions are welcome." and on "Submit a LoRA adapter on the approved base." --
# a guard that patrols the four spellings the last incident happened to use.
#
# SUBMISSION_CLI.md keeps a `### Adapter-only BYO` section whose body says "Withdrawn", because
# outside links point at that anchor; that section carries a marker and stays green. A LINK whose
# target is the withdrawn anchor is an offender on its own -- a withdrawal cannot produce one.

#: Policy documents that are not in DOC_FILES (the other guards read numbers and paths these two
#: do not carry) but which state participant-facing rules. Same fail-closed rule.
POLICY_DOCS = [
    "docs/ARTIFACT-POLICY.md",
    "docs/TRAINING-POLICY.md",
]

#: Directories whose markdown is unit content, not participant-facing prose.
_BYO_SKIP_DIRS = {"units", "units_archived", "corpus", ".git", ".github"}

#: A link whose target is the withdrawn section. The heading itself is not a link, so the
#: withdrawal stub does not match; `[text](SUBMISSION_CLI.md#adapter-only-byo)` does.
_BYO_LINK_RE = re.compile(r"\]\([^)]*#adapter-only-byo\)", re.I)

#: Raises a participant-supplied model path. ``adaptation`` does not match ``adapter``.
_BYO_TERM_RE = re.compile(r"\bBYO\b|\bbyo-(?:small|large)\b|bring[-\s]your[-\s]own|\bLoRA\b|(?<![_a-z])adapters?\b", re.I)

#: The withdrawal, in the spellings the ruling and the toolkit actually use. Deliberately narrow:
#: an unrelated "invalid" or "held" elsewhere in the sentence must NOT clear a BYO offer, which is
#: how the first version let ``Use `byo-large`; an invalid descriptor is rejected`` through.
_BYO_WITHDRAWAL_RES = [
    re.compile(r"not\s+part\s+of\s+this\s+competition", re.I),
    re.compile(r"\bwithdrawn\b", re.I),
    re.compile(r"no\s+longer\s+accepts?", re.I),
    re.compile(r"(?:are|is)\s+invalid|invalid\s+since", re.I),
    re.compile(r"has\s+not\s+accepted", re.I),
    re.compile(r"held\s+by\s+the\s+organi[sz]er", re.I),
    re.compile(r"there\s+is\s+no\s+(?:LoRA|adapter)", re.I),
    re.compile(r"not\s+(?:authorized|permitted|available)", re.I),
]

#: Sentence boundary: a terminator, or a blank line / heading / table row / list bullet, because
#: prose here is hard-wrapped and a sentence routinely spans lines. A colon is NOT a boundary:
#: the withdrawal is regularly the lead-in and the detail follows it ("...are not part of this
#: competition (ruling of 2026-09-18): no LoRA adapter path, no in-image model weights path").
_BYO_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n|\n(?=\s*(?:[#|*+-]|\d+\.))")
#: `>` is deliberately absent above: a hard-wrapped blockquote is one sentence per several
#: `> ` lines, and splitting on it separated a withdrawal lead-in from its own detail.


def _byo_patrolled_docs() -> list[tuple[str, str]]:
    """DOC_FILES and POLICY_DOCS by name (fail-closed), plus every other markdown file.

    Discovered rather than listed for the rest: the residue sat in two documents that no guard
    named, so a list is the failure mode this guard exists to prevent. A new document is
    patrolled the day it lands.
    """
    _require(DOC_FILES, "patrolled document(s)")
    _require(POLICY_DOCS, "policy document(s)")
    rels = list(dict.fromkeys(DOC_FILES + POLICY_DOCS))
    for path in sorted(REPO.rglob("*.md")):
        parts = path.relative_to(REPO).parts
        if any(part in _BYO_SKIP_DIRS or part.startswith(".") for part in parts[:-1]):
            continue
        rel = path.relative_to(REPO).as_posix()
        if rel not in rels:
            rels.append(rel)
    return [(rel, (REPO / rel).read_text(encoding="utf-8")) for rel in rels]


def _byo_offer_offenders(docs: list[tuple[str, str]]) -> list[str]:
    offenders = []
    for rel, text in docs:
        for match in _BYO_LINK_RE.finditer(text):
            offenders.append(f"{rel}: link to the withdrawn section -- {match.group(0)}")
        chunks = [" ".join(chunk.split()) for chunk in _BYO_SENTENCE_RE.split(text)]
        for index, flat in enumerate(chunks):
            if not flat or not _BYO_TERM_RE.search(flat):
                continue
            # A heading names its section; the withdrawal is the sentence under it, so read the
            # two together. `### Adapter-only BYO` over a body that says "Withdrawn" is not an
            # offer, and a heading that offers with no withdrawal below it still fails.
            scope = flat
            if flat.startswith("#"):
                scope = " ".join(filter(None, chunks[index : index + 2]))
            if any(pattern.search(scope) for pattern in _BYO_WITHDRAWAL_RES):
                continue
            offenders.append(f"{rel}: {flat[:120]!r}")
    return offenders


def test_no_document_offers_a_bring_your_own_model_path() -> None:
    offenders = _byo_offer_offenders(_byo_patrolled_docs())
    assert not offenders, (
        "participant-facing documents raise a bring-your-own model or adapter path without the "
        "withdrawal, which the ruling of 2026-09-18 made binding (every submission runs against "
        "the House model; the descriptor has not accepted `byo-*` since toolkit 2.4.3):\n"
        + "\n".join(sorted(set(offenders)))
    )


def test_guard5_catches_its_own_exemplars() -> None:
    """The residue lines this guard was written for, and the five a review showed it missing."""
    exemplars = {
        "README firewall paragraph": (
            "BYO model submissions follow the "
            "[adapter-only contract](SUBMISSION_CLI.md#adapter-only-byo)."
        ),
        "artifact policy": (
            "For an approved LoRA submission, the existing one-adapter/base/rank contract "
            "still applies, and the serving route must separately be available."
        ),
        "training policy link": (
            "It does not expand the allowed model or artifact categories: the "
            "[adapter-only BYO contract](../SUBMISSION_CLI.md#adapter-only-byo) "
            "continues to apply."
        ),
        "training policy example": (
            "For example, an approved adapter may be trained offline using a public, "
            "licensed dataset."
        ),
        "provenance asks for adapter revisions": (
            "Identify the data used for calibration, together with the immutable "
            "model/adapter revisions or checksums."
        ),
        "provenance asks for adapter histories": (
            "Use the existing model disclosure and keep base and adapter training histories "
            "clear in the provenance record."
        ),
        "baseline README": (
            "Declare pinned model and adapter versions and their training cutoffs."
        ),
        "starter-pack wording": "Your submission ships only a LoRA adapter, rank <= 64.",
        # The five the review found green on the phrase-list version of this guard.
        "unrelated 'held' clears the mention": (
            "Use `byo-small`; adapters are held to the same limits as api."
        ),
        "unrelated 'invalid' clears the mention": (
            "Use `byo-large`; an invalid descriptor is rejected at intake."
        ),
        "plain offer": "BYO model submissions are welcome.",
        "offer by permission": "You may bring your own model as a LoRA adapter of rank 32.",
        "offer by instruction": "Submit a LoRA adapter on the approved base.",
    }
    for label, doc in exemplars.items():
        assert _byo_offer_offenders([("<exemplar>", doc)]), (
            f"guard 5 does not catch its own exemplar ({label}); it would be cited as proof "
            "of a class it cannot detect"
        )
    # ...and the withdrawal itself, which names the same words, must stay green.
    clean = (
        "### Adapter-only BYO\n\n"
        "Withdrawn. This section described a LoRA-adapter option; by the ruling of 2026-09-18 "
        "bring-your-own models and adapters are not part of this competition, and the "
        "descriptor no longer accepts the `byo-*` categories. Every submission runs against "
        "the House model through `MODEL_ENDPOINT`.\n\n"
        "The former `byo-large` / `byo-small` categories are invalid since toolkit 2.4.3, and "
        "an upload that still carries one is held by the organizer's intake and never run.\n"
    )
    assert not _byo_offer_offenders([("<clean>", clean)])

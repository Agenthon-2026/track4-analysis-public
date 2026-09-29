"""The smoke judge's NUMBERS, not just that it runs.

`test_judge_required.py` proves the smoke factory produces a working judge and stamps
`rankable=False`. Nothing anywhere proved its arithmetic was right — that a fully supported
hypothesis scores 1.0, an unsupported one 0.0, an empty one 0.0, and that tokens of two
characters or fewer are not content.

Equivalent assertions used to cover an earlier lexical judge that was removed when Track 4
collapsed onto one scoring implementation. They are restated here rather than dropped, because
the code they cover is here: dropping them would leave `LexicalSmokeJudge.entail` with no test
of what it computes.

The smoke judge gates nothing on the production path — `build_smoke_verifier` reports its number
and never ranks on it — so this is not a leaderboard guarantee. It is the guarantee that a
participant previewing admissibility locally gets a number that means what the docs say.
"""

from __future__ import annotations

from qfbench2_track_analysis.judge_factory import LexicalSmokeJudge

# `entail` splits on whitespace and does not strip punctuation, so "consensus." and
# "consensus" are DIFFERENT tokens. The supported hypothesis below is therefore built only from
# tokens that appear in the premise exactly as written — getting this wrong is what made the
# first draft of this file fail, and it is worth stating so the next fixture is chosen the same
# way. Premise content tokens (length > 2): synthetic, issuer, reported, quarterly, earnings,
# above, the, published, consensus.
PREMISE = (
    "Synthetic Issuer A reported quarterly earnings above the published consensus."
)
SUPPORTED = "Synthetic Issuer reported quarterly earnings above the published"
UNSUPPORTED = "Municipal bond coupons settle every Thursday"


def test_full_overlap_scores_one() -> None:
    """Every content token of the hypothesis appears in the premise."""
    assert LexicalSmokeJudge().entail(PREMISE, SUPPORTED) == 1.0


def test_no_overlap_scores_zero() -> None:
    """A hypothesis sharing no content token with the premise is unsupported."""
    assert LexicalSmokeJudge().entail(PREMISE, UNSUPPORTED) == 0.0


def test_empty_hypothesis_scores_zero() -> None:
    """An empty hypothesis is not vacuously entailed.

    This is the one with teeth: `all(t in premise for t in [])` is True, so the natural
    implementation of "every token is supported" scores an empty claim 1.0. A citation whose
    hypothesis resolved to nothing would then be perfectly faithful.
    """
    assert LexicalSmokeJudge().entail(PREMISE, "") == 0.0


def test_tokens_of_two_characters_or_fewer_are_not_content() -> None:
    """Short tokens are stopword-like and must not carry entailment on their own."""
    assert LexicalSmokeJudge().entail("xx yy zz", "xx yy") == 0.0

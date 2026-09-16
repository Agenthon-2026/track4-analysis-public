"""Track 4 (analysis) verifier/scorer, in the shape the shared driver requires.

    LEADERBOARD_SORT       : str                       "desc" — higher composite wins
    build_verifier(ctx)    -> HierarchicalVerifier     the OFFICIAL, rankable factory. It
                                                       constructs the pinned production NLI judge
                                                       or raises an organizer fault; there is no
                                                       judge-less path and no perfect default.
    build_smoke_verifier(ctx) -> HierarchicalVerifier  the separately named, always-non-rankable
                                                       local preview factory.
    score_unit(ctx, ...)   -> UnitOutcome              the single scoring implementation every
                                                       Track-4 entrypoint (public verifier, smoke
                                                       preview, sealed final scorer) goes through.

The driver passes exactly ``{unit_dir, output_dir, failure_map}``. Everything trusted — the card,
the task, the entity roster, the question cutoff, the digest-verified corpus index and the resolved
outcome when the phase supplies one — is hydrated from ``unit_dir`` and from any plan-derived
values the caller injects, which take precedence over the mounted tree.
"""

from .scoring import (
    LEADERBOARD_SORT,
    SCORER_VERSION,
    UnitOutcome,
    build_smoke_verifier,
    build_verifier,
    score_unit,
    scorer_identity,
)

__all__ = [
    "LEADERBOARD_SORT",
    "SCORER_VERSION",
    "UnitOutcome",
    "build_smoke_verifier",
    "build_verifier",
    "score_unit",
    "scorer_identity",
]

# The shared scorer version (see scoring.SCORER_VERSION). Exposed so a
# participant can read it without importing the scoring module.
__version__ = SCORER_VERSION

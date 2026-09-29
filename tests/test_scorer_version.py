"""The version a participant resolves must be the version the package ships.

Track 2 previously declared 2.1.0 in `pyproject.toml` and 2.0.0 in the package, so a participant who
found a version could not trust it. This pins the two declarations in this repository to each other,
and pins the Track 4 scorer version itself. Track 4's scorer version now moves on its own, apart from
the other tracks' scorers.
"""

from __future__ import annotations

import pathlib
import re

from qfbench2_track_analysis.scoring import SCORER_VERSION, scorer_identity

PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"


def _declared() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M)
    assert m, "pyproject.toml declares no version"
    return m.group(1)


def test_pyproject_and_scorer_version_agree():
    assert _declared() == SCORER_VERSION, (
        f"pyproject.toml says {_declared()} and SCORER_VERSION says {SCORER_VERSION}; "
        "a participant who found one of them could not trust it"
    )


def test_the_track4_scorer_version_is_pinned():
    """Track 4's scorer version now moves on its own; it is no longer shared with the other tracks.

    Changing it is a scoring change that participants are told about, so a bump must update this pin
    on purpose.
    """
    assert SCORER_VERSION == "5.2.0", (
        f"SCORER_VERSION is {SCORER_VERSION}; the published Track 4 scorer version is 5.2.0. "
        "Update this pin only together with a published scoring notice."
    )


def test_scorer_identity_carries_the_version():
    ident = scorer_identity()
    assert ident["scorer_version"] == SCORER_VERSION
    assert ident["scorer_package"].endswith(".scoring")


def test_the_package_itself_exposes_the_version():
    """A participant reads `qfbench2_track_analysis.__version__`, not the scoring module's constant.

    Track 4 shipped `SCORER_VERSION` in `scoring` and re-exported it, but exposed no
    `__version__` and no `scorer_identity` on the package -- and the tests above could not see
    that, because they import from `.scoring` directly and bypass the re-export entirely.
    """
    import qfbench2_track_analysis

    assert qfbench2_track_analysis.__version__ == SCORER_VERSION
    assert qfbench2_track_analysis.scorer_identity() == scorer_identity()

"""`templates/card.toml` must not become a second source of truth.

The authoring template was removed from this repository once because prose could agree with it
and still disagree with every card actually shipped. It is published again as an annotated
example — participants asked for the whole shape in one place — but only under a guard: the
scoring parameters, the compute grant and the agent budget it shows are pinned to the exemplar
unit's card, which is the card a participant actually receives.

Standard library only, so this runs in the secret-free ``firewall`` CI job.

Fail-closed: if either file is missing, this test FAILS. It does not skip.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "templates" / "card.toml"
EXEMPLAR = REPO / "units" / "t4-EXAMPLE-eps-beat" / "card.toml"

#: Blocks whose values must be identical in both files.
PINNED_BLOCKS = [
    ("scoring", "params"),
    ("environment",),
    ("agent",),
]

#: Keys inside a pinned block that are legitimately per-unit and are NOT compared.
PER_UNIT_KEYS = {("scoring", "params"): {"target_type"}}


def _load(path: Path) -> dict:
    assert path.exists(), (
        f"{path.relative_to(REPO)} is missing. This guard pins the published template to the "
        "exemplar card; a missing file means the pin stopped existing, not that it stopped "
        "mattering (the fail-closed rule)."
    )
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _block(card: dict, keys: tuple[str, ...]) -> dict:
    node: dict = card
    for key in keys:
        assert key in node, f"card has no [{'.'.join(keys)}] block"
        node = node[key]
    return node


def test_template_card_pins_the_exemplars_scoring_and_environment() -> None:
    template, exemplar = _load(TEMPLATE), _load(EXEMPLAR)
    offenders = []
    for keys in PINNED_BLOCKS:
        t_block, e_block = _block(template, keys), _block(exemplar, keys)
        skip = PER_UNIT_KEYS.get(keys, set())
        for name in sorted(set(t_block) | set(e_block)):
            if name in skip:
                continue
            if t_block.get(name) != e_block.get(name):
                offenders.append(
                    f"[{'.'.join(keys)}] {name}: template={t_block.get(name)!r} "
                    f"exemplar={e_block.get(name)!r}"
                )
    assert not offenders, (
        "templates/card.toml has drifted from the exemplar unit's card. The template is an "
        "illustration; the exemplar is what a participant receives. Make them agree:\n"
        + "\n".join(offenders)
    )


def test_template_is_marked_as_non_authoritative() -> None:
    """A reader must not mistake the illustration for the contract."""
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "NOT authoritative" in text
    assert "units/t4-EXAMPLE-eps-beat/card.toml" in text


def test_the_guard_would_notice_a_drifted_template() -> None:
    """The control: a template whose weights are inverted must be caught."""
    template, exemplar = _load(TEMPLATE), _load(EXEMPLAR)
    drifted = {
        "scoring": {"params": dict(template["scoring"]["params"])},
        "environment": dict(template["environment"]),
        "agent": dict(template["agent"]),
    }
    weights = list(drifted["scoring"]["params"]["composite_weights"])
    drifted["scoring"]["params"]["composite_weights"] = list(reversed(weights))
    assert (
        drifted["scoring"]["params"]["composite_weights"]
        != exemplar["scoring"]["params"]["composite_weights"]
    ), "the control does not differ from the exemplar, so it proves nothing"

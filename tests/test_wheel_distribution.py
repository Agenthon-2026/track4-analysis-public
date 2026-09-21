"""Exercise the installed wheel, without the checkout supplying missing runtime modules."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    scratch = tmp_path_factory.mktemp("wheel-distribution")
    source = scratch / "source"
    source.mkdir()
    shutil.copyfile(root / "pyproject.toml", source / "pyproject.toml")
    shutil.copyfile(root / "LICENSE", source / "LICENSE")
    for name in ("qfbench2_track_analysis", "faithfulness"):
        shutil.copytree(root / name, source / name, ignore=shutil.ignore_patterns("__pycache__"))
    wheels = scratch / "wheels"
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "--no-index", "--wheel-dir", str(wheels), str(source)],
        check=True, capture_output=True, text=True,
    )
    wheel, = wheels.glob("*.whl")
    target = scratch / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
         "--target", str(target), str(wheel)],
        check=True, capture_output=True, text=True,
    )
    return scratch, target, wheel


def test_wheel_contains_runtime_packages_without_source_only_content(installed_wheel):
    _, _, wheel = installed_wheel
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert {"faithfulness/__init__.py", "faithfulness/judge.py",
            "qfbench2_track_analysis/judge_factory.py"} <= names
    assert not any("/tests/" in name or "/serving/" in name for name in names)
    assert all(name.startswith(("faithfulness/", "qfbench2_track_analysis/",
                                "qfbench2_track_analysis-")) for name in names)
    assert all(name.endswith(".py") or ".dist-info/" in name for name in names)


def test_installed_production_factory_imports_and_refuses_unavailable_models(installed_wheel):
    scratch, target, _ = installed_wheel
    probe = r'''
import json, sys
from pathlib import Path
target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
import faithfulness.judge as judge
import qfbench2_track_analysis.judge_factory as factory
import qfbench2_track_analysis.scoring as scoring
from qfbench2_track_analysis.codes import T4OrganizerFault
for module in (judge, factory, scoring):
    assert Path(module.__file__).resolve().is_relative_to(target), module.__file__
assert scoring._SCHEMA.is_file()
try:
    factory.build_production_judge()
except T4OrganizerFault as exc:
    assert 'no judge artifact' in str(exc)
else:
    raise AssertionError('unconfigured production judge did not refuse')
spec = factory.JudgeSpec(
    ('synthetic/missing-model',), {'synthetic/missing-model': '1' * 40},
    'sha256:' + '2' * 64, 'sha256:' + '3' * 64, str(Path.cwd() / 'missing-cache'))
try:
    factory.build_production_judge(spec)
except T4OrganizerFault as exc:
    assert 'not a directory' in str(exc)
else:
    raise AssertionError('missing model cache did not refuse')
# Reach the actual installed default builder without weights or any network attempt.
# Missing runtime is an independent organizer fault, never a lexical smoke fallback.
cache = Path.cwd() / 'synthetic-cache'
cache.mkdir()
(cache / 'synthetic.txt').write_text('not model weights')
spec = factory.JudgeSpec(spec.model_ids, spec.model_revisions, spec.tokenizer_digest,
                         factory.compute_cache_tree_digest(cache), str(cache))
judge._TRANSFORMERS_AVAILABLE = False
try:
    factory.build_production_judge(spec)
except T4OrganizerFault as exc:
    assert type(exc.__cause__) is ImportError, repr(exc.__cause__)
else:
    raise AssertionError('missing runtime did not refuse')
print(json.dumps({'installed_factory_import': True, 'fail_closed_controls': 3}))
'''
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("QFBENCH2_T4_")}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(target)],
        cwd=scratch, env=environment, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == {
        "installed_factory_import": True, "fail_closed_controls": 3}

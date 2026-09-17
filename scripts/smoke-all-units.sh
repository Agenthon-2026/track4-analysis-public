#!/usr/bin/env bash
# Run baseline_agent + faithfulness judge + qfbench2-smoke on all public T4 units.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "${ROOT}/.venv/bin/activate"
export PYTHONPATH="$ROOT"
OUT="${1:-/tmp/t4-baseline}"
mkdir -p "$OUT/units"

pass_gate=0
pass_smoke=0
total=0
failures=0

for unit_dir in units/t4-*/; do
  unit_id="$(basename "$unit_dir")"
  unit_out="$OUT/units/$unit_id"
  mkdir -p "$unit_out"
  ans="$unit_out/answer.json"
  total=$((total + 1))

  echo "== $unit_id =="
  python baselines/baseline_agent.py \
    --task "$unit_dir/task.json" \
    --corpus "$unit_dir/corpus" \
    --out "$ans"

  if python faithfulness/judge.py --answer "$ans" --unit "$unit_dir" >/dev/null 2>&1; then
    pass_gate=$((pass_gate + 1))
    echo "  faithfulness: PASS"
  else
    failures=$((failures + 1))
    echo "  faithfulness: FAIL"
  fi

  if qfbench2-smoke "$unit_dir" "$unit_out" --track analysis >/dev/null 2>&1; then
    pass_smoke=$((pass_smoke + 1))
    echo "  smoke: PASS"
  else
    failures=$((failures + 1))
    echo "  smoke: FAIL"
  fi
done

echo ""
echo "faithfulness gate: $pass_gate/$total"
echo "smoke admissible:  $pass_smoke/$total"

if (( failures > 0 )); then
  exit 1
fi

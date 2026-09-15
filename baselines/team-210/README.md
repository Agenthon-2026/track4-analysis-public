# Team 210 Analysis Agent Baseline

This directory packages the Team 210 participant analysis baseline for Track 4 (Evidence-Grounded Tabular Analysis).

## Architecture

1. **CLI Contract**: Implements `analyze --task <path> --corpus <path> --out <path>` conforming to `interface_version = "2.0"`.
2. **Schema Compliance**: Emits predictions and confidence intervals strictly adhering to `analysis.schema.json`.
3. **Faithfulness & Citation Engine**: Matches text-corpus citations to entity predictions, optimizing for both task accuracy and grounding faithfulness.
4. **Network**: Restricted model endpoint access via organizer proxy with audited egress.

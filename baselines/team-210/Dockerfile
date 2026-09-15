# Agenthon 2026 Track 4 submission image.
#
# Stdlib only. Retrieval is BM25 over sentence spans and the validators are
# plain Python, so the floor runs with no model at all - which matters because
# the endpoint is the one dependency that can be slow, metered or refused, and a
# Track 4 submission whose citations never get written is inadmissible rather
# than merely inaccurate.
FROM python:3.13-slim

WORKDIR /agent
COPY agent.py /agent/agent.py
COPY analysis_core/ /agent/analysis_core/

LABEL qfbench2.interface_version="2.0"

ENTRYPOINT ["python", "/agent/agent.py"]
CMD ["analyze", "--task", "/input/task.json", "--corpus", "/input/corpus/", \
     "--out", "/output/answer.json"]

"""Tests for profiling utilities."""

import json
from pathlib import Path

from profiling.profiler import TraceAggregator


def test_trace_aggregator_merges_traces(tmp_path):
    rank0 = tmp_path / "profile_rank0.json"
    rank1 = tmp_path / "profile_rank1.json"

    rank0.write_text(json.dumps({"events": ["a"]}))
    rank1.write_text(json.dumps({"events": ["b"]}))

    aggregator = TraceAggregator(output_dir=str(tmp_path), world_size=2)
    out_path = aggregator.aggregate("merged.json")

    merged = json.loads(Path(out_path).read_text())
    assert len(merged["traces"]) == 2
    assert merged["traces"][0] == {"events": ["a"]}

"""Tests for telemetry agent components."""

import importlib
from pathlib import Path

import pytest
import torch
from prometheus_client import registry as prometheus_registry


def _metric_value(name: str, labels: dict[str, str]) -> float | None:
    return prometheus_registry.REGISTRY.get_sample_value(name, labels)


@pytest.fixture
def telemetry_module(fresh_prometheus_registry):
    import sys
    from prometheus_client import Gauge, Counter
    import prometheus_client

    class MetricValue:
        def __init__(self):
            self._value = 0.0

        def set(self, value):
            self._value = float(value)

        def observe(self, value):
            self.set(value)

        def inc(self, value=1.0):
            self._value += float(value)

        def get(self):
            return self._value

    class DummyMetric:
        def __init__(self, *args, **kwargs):
            self._metrics = {}

        def labels(self, **labels):
            key = tuple(sorted(labels.items()))
            if key not in self._metrics:
                wrapper = MetricValue()

                class Wrapper:
                    def __init__(self, metric_value):
                        self._value = metric_value

                    def set(self, value):
                        self._value.set(value)

                    def observe(self, value):
                        self._value.observe(value)

                    def inc(self, value=1.0):
                        self._value.inc(value)

                self._metrics[key] = Wrapper(wrapper)
            return self._metrics[key]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(prometheus_client, "Gauge", DummyMetric)
    monkeypatch.setattr(prometheus_client, "Counter", DummyMetric)
    monkeypatch.setattr(prometheus_client, "Histogram", DummyMetric)

    sys.modules.pop("observability.telemetry_agent", None)
    module = importlib.import_module("observability.telemetry_agent")
    monkeypatch.undo()
    return module


def test_nccl_trace_parser_updates_metric(tmp_path, telemetry_module):
    log_file = tmp_path / "nccl.log"
    log_file.write_text("AllReduce op latency 6 4\n")

    parser = telemetry_module.NCCLTraceParser(str(log_file), rank=0)
    telemetry_module.nccl_imbalance.labels(rank=parser.rank).set(0.0)

    detected = parser.parse()
    assert detected == 1

    value = telemetry_module.nccl_imbalance.labels(rank=parser.rank)._value.get()
    assert value > 0


def test_training_telemetry_updates_gauges(telemetry_module):
    telemetry = telemetry_module.TrainingTelemetry(rank=2)
    grad_tensor = torch.ones(4)

    telemetry.log_step(loss=1.5, tokens=200, duration=4.0, grad_tensor=grad_tensor)

    loss_value = telemetry_module.loss_metric.labels(rank=2)._value.get()
    throughput_value = telemetry_module.throughput.labels(rank=2)._value.get()
    grad_value = telemetry_module.grad_norm.labels(rank=2)._value.get()

    assert loss_value == pytest.approx(1.5)
    assert throughput_value == pytest.approx(50.0)
    assert grad_value == pytest.approx(torch.norm(grad_tensor).item())

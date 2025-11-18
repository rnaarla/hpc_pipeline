"""Tests for benchmarking.roofline_analysis."""

import torch
import pytest

from benchmarking.roofline_analysis import fit_kaplan, run_roofline


def test_run_roofline_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    flops, bandwidth, intensity = run_roofline()

    assert flops == 0.0
    assert bandwidth == 0.0
    assert intensity == 0.0


def test_fit_kaplan_returns_parameters():
    data = [
        (1e8, 1e9, 2.5),
        (2e8, 3e9, 2.2),
        (5e8, 1e10, 1.9),
        (1e9, 1e10, 1.7),
    ]
    params = fit_kaplan(data)

    assert len(params) == 5
    assert all(isinstance(p, float) for p in params)

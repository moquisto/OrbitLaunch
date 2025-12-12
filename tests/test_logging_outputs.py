import csv
import numpy as np
import types
import tempfile
import os

import importlib
import main
import pytest

from Logging import generate_logs # Import generate_logs directly


otw_module = importlib.import_module("Analysis.optimization")


def test_log_iteration_writes_row(tmp_path, monkeypatch):
    log_file = tmp_path / "twostage_log.csv"
    monkeypatch.setattr(generate_logs, "LOG_FILENAME", str(log_file))

    # minimal params and results
    params = np.arange(35, dtype=float)
    # We need to create an OptimizationParams object for log_iteration
    from Analysis.config import OptimizationParams
    params_obj = OptimizationParams(*params)

    results = {"cost": 1.23, "fuel": 4.56, "orbit_error": 7.89, "status": "OK"}

    # Ensure file exists with header
    with open(generate_logs.LOG_FILENAME, "w", newline="") as f:
        csv.writer(f).writerow(["phase"])

    generate_logs.log_iteration("PhaseX", 1, params_obj, results)

    with open(log_file, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2
    assert rows[-1][0] == "PhaseX"
    assert rows[-1][-1] == "OK"


@pytest.mark.skip("optimize_logging module removed; legacy optimizer deprecated")
def test_optimize_logging_objective_writes_csv(tmp_path, monkeypatch):
    pass

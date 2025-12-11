import csv
import numpy as np
import types
import tempfile
import os

import importlib
import main
import pytest

# Ensure main exposes constants expected by optimizer modules (they may not be re-exported).
if not hasattr(main, "MU_EARTH"):
    from gravity import MU_EARTH as _MU
    main.MU_EARTH = _MU
if not hasattr(main, "R_EARTH"):
    from gravity import R_EARTH as _RE
    main.R_EARTH = _RE

otw = importlib.import_module("optimization_twostage")


def test_log_iteration_writes_row(tmp_path, monkeypatch):
    log_file = tmp_path / "twostage_log.csv"
    monkeypatch.setattr(otw, "LOG_FILENAME", str(log_file))

    # minimal params and results
    params = np.arange(35, dtype=float)
    results = {"cost": 1.23, "fuel": 4.56, "orbit_error": 7.89, "status": "OK"}

    # Ensure file exists with header
    with open(otw.LOG_FILENAME, "w", newline="") as f:
        csv.writer(f).writerow(["phase"])

    otw.log_iteration("PhaseX", 1, params, results)

    with open(log_file, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2
    assert rows[-1][0] == "PhaseX"
    assert rows[-1][-1] == "OK"


@pytest.mark.skip("optimize_logging module removed; legacy optimizer deprecated")
def test_optimize_logging_objective_writes_csv(tmp_path, monkeypatch):
    pass

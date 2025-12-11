import csv
import numpy as np
import types
import tempfile
import os

import importlib
import main

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
    params = np.arange(29, dtype=float)
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


def test_optimize_logging_objective_writes_csv(tmp_path, monkeypatch):
    optlog = importlib.import_module("optimize_logging")
    log_file = tmp_path / "opt_log.csv"
    monkeypatch.setattr(optlog, "LOG_FILENAME", str(log_file))

    # Reinitialize header for this test
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["iteration", "status"])

    # Stub simulation to avoid heavy run
    state0 = types.SimpleNamespace(m=100.0)
    dummy_log = types.SimpleNamespace(
        m=[90.0],
        r=[np.array([optlog.R_EARTH + optlog.TARGET_ALT_M, 0, 0])],
        v=[np.array([0, np.sqrt(optlog.MU_EARTH / (optlog.R_EARTH + optlog.TARGET_ALT_M)), 0])],
    )
    monkeypatch.setattr(optlog, "build_simulation", lambda: (types.SimpleNamespace(run=lambda *a, **k: dummy_log), state0, 0.0))
    monkeypatch.setattr(optlog, "orbital_elements_from_state", lambda r, v, mu: (0.0, optlog.R_EARTH + optlog.TARGET_ALT_M, optlog.R_EARTH + optlog.TARGET_ALT_M))

    cost = optlog.objective_function([5.0, 2.0, 80.0, 0.9])
    assert cost > 0

    with open(log_file, newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2  # header + one log row
    assert rows[-1][-1] in ("PERFECT", "GOOD", "OK")

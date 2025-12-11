import numpy as np

from main import main_orchestrator
from Environment.config import EnvironmentConfig
from Hardware.config import HardwareConfig
from Software.config import SoftwareConfig
from Main.config import SimulationConfig
from Logging.config import LoggingConfig
from Analysis.config import AnalysisConfig


def test_build_simulation_with_parameterized_pitch(monkeypatch):
    # Instantiate all config objects
    env_config = EnvironmentConfig()
    hw_config = HardwareConfig()
    sw_config = SoftwareConfig()
    sim_config = SimulationConfig()
    log_config = LoggingConfig()
    analysis_config = AnalysisConfig()

    sw_config.pitch_guidance_mode = "parameterized"

    # Call main_orchestrator to get the sim object
    sim, state0, t0, log_cfg, analysis_cfg = main_orchestrator(
        env_config=env_config,
        hw_config=hw_config,
        sw_config=sw_config,
        sim_config=sim_config,
        log_config=log_config,
        analysis_config=analysis_config
    )

    direction = sim.guidance.pitch_program(0.0, state0)
    assert np.isclose(np.linalg.norm(direction), 1.0)
    # Should align with radial direction for the default simple program
    r_hat = state0.r_eci / np.linalg.norm(state0.r_eci)
    # The default pitch program starts at 89.8 degrees, which is almost vertical
    assert np.isclose(np.dot(direction, r_hat), np.sin(np.radians(89.8)), atol=1e-3)


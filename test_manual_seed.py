import numpy as np
import copy
from main import main_orchestrator
from Environment.config import EnvironmentConfig
from Hardware.config import HardwareConfig
from Software.config import SoftwareConfig
from Main.config import SimulationConfig
from Logging.config import LoggingConfig
from Analysis.config import AnalysisConfig, OptimizationParams
from Analysis.cost_functions import evaluate_simulation_results, PENALTY_CRASH
from Software.guidance import configure_software_for_optimization

# Set the global optimization phase for testing
current_optimization_phase = 1

def run_simulation_wrapper(params: OptimizationParams, env_config: EnvironmentConfig, hw_config: HardwareConfig, sw_config: SoftwareConfig, sim_config: SimulationConfig, log_config: LoggingConfig):
    if not isinstance(params, OptimizationParams):
        params = OptimizationParams(*params)
    
    cfg_env = copy.deepcopy(env_config)
    cfg_hw = copy.deepcopy(hw_config)
    cfg_sw = copy.deepcopy(sw_config)
    cfg_sim = copy.deepcopy(sim_config)
    cfg_log = copy.deepcopy(log_config)

    cfg_sw, cfg_sim = configure_software_for_optimization(params, cfg_sw, cfg_sim, cfg_env)

    results = {"fuel": 0.0, "orbital_error": PENALTY_CRASH, "status": "INIT", "cost": PENALTY_CRASH}
    
    try:
        sim, state0, t0, _log_config, _analysis_config = main_orchestrator(
            env_config=cfg_env,
            hw_config=cfg_hw,
            sw_config=cfg_sw,
            sim_config=cfg_sim,
            log_config=cfg_log,
        )
        initial_mass = state0.m

        log = sim.run(t0, duration=10000.0, dt=1.0, state0=state0)
        max_altitude = max(log.altitude) if log.altitude else 0.0

        results = evaluate_simulation_results(log, initial_mass, cfg_env, cfg_sim, max_altitude, phase=current_optimization_phase)

    except Exception:
        results["status"] = "SIM_FAIL_UNKNOWN"
        import traceback
        traceback.print_exc()
    finally:
        results["error"] = results.get("orbital_error", PENALTY_CRASH)
        results["orbit_error"] = results["error"]

    return results

if __name__ == "__main__":
    env_config = EnvironmentConfig()
    hw_config = HardwareConfig()
    sw_config = SoftwareConfig()
    sim_config = SimulationConfig()
    log_config = LoggingConfig()
    analysis_config = AnalysisConfig()

    manual_seed_params = analysis_config.optimizer_manual_seed
    if not manual_seed_params:
        print("Manual seed not found or is empty in AnalysisConfig. Exiting.")
        exit()

    params_obj = OptimizationParams(*manual_seed_params)

    print(f"Testing manual seed with Phase: {current_optimization_phase}")
    test_results = run_simulation_wrapper(params_obj, env_config, hw_config, sw_config, sim_config, log_config)

    print("\n--- Manual Seed Test Results ---")
    print(f"Fuel Used: {test_results['fuel']:.1f} kg")
    print(f"Error: {test_results['error']:.1f} m ({test_results['error']/1000:.1f} km)")
    print(f"Status: {test_results['status']}")
    if "perigee_alt_m" in test_results:
        print(f"Perigee Altitude: {test_results['perigee_alt_m']:.1f} m")
    if "eccentricity" in test_results:
        print(f"Eccentricity: {test_results['eccentricity']:.4f}")

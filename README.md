# OrbitLaunch

Object-oriented 3D simulation of a two-stage rocket (BFR-inspired) launching from Earth to a target circular orbit. The goal is to search for a guidance profile that reaches a stable orbit at a chosen altitude (default: ~420 km LEO) while minimizing propellant burned.

If you later draw a 2D cross-section of the orbit, the circular path will appear to pass through Earth's center because the orbital plane passes through the center; the orbit itself is always above the surface.

## Repo layout
- **Environment/**: Earth model, atmosphere (`ussa1976` + `pymsis`), aerodynamics, and orbital element helpers.
- **Hardware/**: Two-stage rocket model + engine parameters.
- **Software/**: Guidance + event/staging logic.
- **Main/**: Integrators and simulation loop.
- **Analysis/**: Cost functions, plotting helpers, and the two-stage optimizer (`Analysis/optimization.py`).
- **Logging/**: CSV logging utilities (optimizer logs + trajectory logs).

## Configuration
Configuration is split into dataclass configs per subsystem:
- `Environment/config.py`, `Hardware/config.py`, `Software/config.py`, `Main/config.py`, `Logging/config.py`
- Optimization parameterization and bounds live in `Analysis/config.py`

## Running

### Install dependencies
This project expects Python 3 and the following packages:
- simulation: `numpy`, `ussa1976`, `pymsis`
- optimization: `scipy` (optional faster optimizer: `cma`)
- for plotting: `matplotlib`
- optional (tests): `pytest`

Example:
```bash
python3 -m pip install numpy scipy ussa1976 pymsis matplotlib cma pytest
```

### Run a single simulation
```bash
python3 main.py
```

### Run the two-stage optimizer (direct insertion)
```bash
python3 Analysis/optimization.py
```

Outputs:
- Optimizer CSV log: `optimization_twostage_log.csv` (override with `ORBITLAUNCH_LOG_FILENAME`)
  - If the CSV header changes, the previous file is auto-moved to `optimization_twostage_log_legacy_<timestamp>.csv`.
- Final plotted trajectory log (same data as the final plot): `trajectory_plots/final_trajectory_log.csv` (disable with `ORBITLAUNCH_SAVE_FINAL_TRAJ_LOG=0`)

## Orbit evaluation (no “cheating”)
Orbit quality is computed from the simulation’s state (ECI position/velocity), not by post-processing the trajectory into a better orbit.

- The optimizer evaluates orbit at a single “evaluation” index near end-of-burn (or at max altitude for impact cases).
- It computes osculating Kepler elements (two-body) from that state to get perigee/apoapsis.
- The reported orbit error is `max(|perigee-target|, |apoapsis-target|)` so both must be within tolerance.

### Direct insertion cutoff
By default the simulation commands throttle to zero once both perigee and apoapsis are within tolerance:
- `ORBITLAUNCH_DIRECT_INSERTION_CUTOFF=1` (default)
- `ORBITLAUNCH_DIRECT_INSERTION_TOL_M=10000` (±10 km)

## Useful environment variables
- Optimizer iterations: `ORBITLAUNCH_PHASE1_MAXITER`, `ORBITLAUNCH_PHASE1_POPSIZE`, `ORBITLAUNCH_PHASE2_COARSE_MAXITER`, `ORBITLAUNCH_PHASE2_COARSE_POPSIZE`, `ORBITLAUNCH_PHASE2_POLISH_MAXITER`, `ORBITLAUNCH_PHASE2_POLISH_POPSIZE`
- Timesteps: `ORBITLAUNCH_PHASE2_POLISH_DT_S`, `ORBITLAUNCH_FINAL_DT_S`, `ORBITLAUNCH_FINAL_EVAL_DT_S`, `ORBITLAUNCH_FINAL_DURATION_S`
- Plot/log outputs: `ORBITLAUNCH_PLOT_FINAL`, `ORBITLAUNCH_ANIMATE_FINAL`, `ORBITLAUNCH_SAVE_FINAL_TRAJ_LOG`, `ORBITLAUNCH_FINAL_TRAJ_LOG_FILE`
- What gets optimized: `ORBITLAUNCH_OPTIMIZE_UPPER_THROTTLE`, `ORBITLAUNCH_OPTIMIZE_BOOSTER_THROTTLE`
- Upper-stage pitch behavior: `ORBITLAUNCH_UPPER_HOLD_LAST_PITCH_THROUGH_BURN`

## Simplifications (v1)
To enable a focused study on rocket guidance and orbital mechanics, and to manage computational complexity, several simplifications have been made in this simulation. These choices allow for faster iteration and highlight the core physics relevant to achieving orbit, rather than getting bogged down in minute details. The primary goal is to provide a robust framework for testing guidance profiles and optimization strategies.

Environment and gravity  
1. Spherical Earth with central gravity. J2 perturbation can be enabled.
2. No third-body perturbations: Moon, Sun, and other bodies are ignored.  
3. Uniform, constant Earth rotation: constant angular velocity vector; no precession, nutation, or tides.

Atmosphere and aerodynamics  
4. Layered atmosphere: US Standard Atmosphere 1976 up to ~86 km, NRLMSIS 2.1 above.  
5. Limited variability: no horizontal variation; time dependence fixed/averaged.  
6. Co-rotating atmosphere with a simple jet stream model.
7. Drag-only aerodynamics: no lift, side force, or moments.  
8. Simplified drag coefficient and reference area: Cd from a Mach curve; one reference area per configuration.

Vehicle and propulsion  
9. Rigid point-mass translational dynamics (3-DOF); no structural flexibility.  
10. No rotational dynamics / ideal attitude control; thrust vector aligns instantly with commands.  
11. Simplified engine model: thrust from prescribed level and Isp via mdot = -T / (Isp * g0).  
12. Simple Isp–pressure relation: linear interpolation between sea-level and vacuum.
13. Ideal, instantaneous staging: mass and reference area switch instantly; no separation dynamics.  
14. Approximate BFR-like parameters; not a detailed replica.

Guidance, control, and constraints  
15. Deterministic, perfect guidance: commands followed exactly; no sensor noise or estimation.  
16. No failures or off-nominal events: no engine-out or aborts.  
17. Simplified constraint treatment: simple checks/penalties for max q and max axial acceleration.

Numerics and optimization  
18. Fixed-step time integration (RK4 or Velocity Verlet).
19. Deterministic simulations only; no Monte Carlo.  
20. Parameterized guidance (pitch/throttle parameters) instead of full optimal control.

Orbit target and stability notion  
21. Orbit quality is evaluated from osculating (two-body) perigee/apoapsis computed from the simulated state near end-of-burn, targeting r = R_E + h_target.  
22. Stability defined in the ideal two-body model: long-term J2, drag, and third bodies are neglected when defining stability.

## Next steps
- Refine guidance parameterization (more late-burn control / shaping).
- Add more detailed plotting/analysis tools and diagnostics.
- Extend constraints and/or add higher-fidelity environmental effects as needed.

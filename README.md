# OrbitLaunch

Object-oriented 3D simulation of a two-stage rocket (BFR-inspired) launching from Earth to a target circular orbit. The goal is to search for a guidance profile that reaches a stable orbit at a chosen altitude while minimizing fuel consumption.

If you later draw a 2D cross-section of the orbit, the circular path will appear to pass through Earth's center because the orbital plane passes through the center; the orbit itself is always above the surface.

## Current structure
- **config.py**: Central configuration file for all simulation parameters.
- **main.py**: Entry point for running a simulation. Includes a simple guidance system and plotting functions.
- **simulation.py**: Core simulation class that integrates all the models and runs the simulation.
- **rocket.py**: Rocket model, including engine performance, staging logic, and throttle schedule.
- **atmosphere.py**: Combined atmosphere model using US Standard Atmosphere 1976 and NRLMSIS 2.1.
- **gravity.py**: Earth gravity model, including the J2 perturbation.
- **aerodynamics.py**: Aerodynamic drag model with a flexible drag coefficient model.
- **integrators.py**: Implementations of RK4 and Velocity Verlet numerical integrators.
- **optimise.py**: Placeholder for the optimization code.

## Configuration
All simulation parameters, including vehicle properties, target orbit, guidance parameters, and physical constants, can be modified in the `config.py` file.

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
21. Circular orbit around spherical Earth: radius r = R_E + h_target in a plane through Earth's center, with correct speed and zero radial velocity at cutoff.  
22. Stability defined in the ideal two-body model: long-term J2, drag, and third bodies are neglected when defining stability.

## Next steps
- Implement optimization algorithms in `optimise.py` to search for optimal guidance profiles.
- Implement more advanced guidance strategies.
- Add more detailed plotting and analysis tools.

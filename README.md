# OrbitLaunch

Object-oriented 3D simulation of a two-stage rocket (BFR-inspired) launching from Earth to a target circular orbit. The goal is to search for a guidance profile that reaches a stable orbit at a chosen altitude while minimizing fuel consumption.

If you later draw a 2D cross-section of the orbit, the circular path will appear to pass through Earth's center because the orbital plane passes through the center; the orbit itself is always above the surface.

## Current structure
- atmosphere.py: US Standard Atmosphere 1976 + NRLMSIS 2.1 placeholders combined by altitude.
- gravity.py: Earth parameters, central gravity, and atmosphere co-rotation velocity.
- aerodynamics.py: Drag-only aerodynamics with a Cd model and reference area query.
- rocket.py: Engine, Stage, and Rocket classes (mass, thrust, staging hooks).
- integrators.py: State dataclass plus RK4 integrator.
- simulation.py: Guidance stub, control command, logger, derivatives, and main loop.

## Simplifications (v1)

Environment and gravity  
1. Spherical Earth with central gravity only: g(r) = -mu * r / |r|^3. Higher-order harmonics are ignored.  
2. No third-body perturbations: Moon, Sun, and other bodies are ignored.  
3. Uniform, constant Earth rotation: constant angular velocity vector; no precession, nutation, or tides.

Atmosphere and aerodynamics  
4. Layered atmosphere: US Standard Atmosphere 1976 up to ~80 km, NRLMSIS 2.1 above.  
5. Limited variability: no horizontal variation; time dependence fixed/averaged.  
6. Co-rotating, windless atmosphere: v_air = omega_E x r_ECI.  
7. Drag-only aerodynamics: no lift, side force, or moments.  
8. Simplified drag coefficient and reference area: Cd constant or simple Mach curve; one reference area per configuration.

Vehicle and propulsion  
9. Rigid point-mass translational dynamics (3-DOF); no structural flexibility.  
10. No rotational dynamics / ideal attitude control; thrust vector aligns instantly with commands.  
11. Simplified engine model: thrust from prescribed level and Isp via mdot = -T / (Isp * g0).  
12. Simple Isp–pressure relation: constant or simple function of ambient pressure.  
13. Ideal, instantaneous staging: mass and reference area switch instantly; no separation dynamics.  
14. Approximate BFR-like parameters; not a detailed replica.

Guidance, control, and constraints  
15. Deterministic, perfect guidance: commands followed exactly; no sensor noise or estimation.  
16. No failures or off-nominal events: no engine-out or aborts.  
17. Simplified constraint treatment: simple checks/penalties for max q and max axial acceleration; no aero-heating model.

Numerics and optimization  
18. Fixed-step time integration initially (RK4, Verlet-like).  
19. Deterministic simulations only; no Monte Carlo.  
20. Parameterized guidance (pitch/throttle parameters) instead of full optimal control.

Orbit target and stability notion  
21. Circular orbit around spherical Earth: radius r = R_E + h_target in a plane through Earth's center, with correct speed and zero radial velocity at cutoff.  
22. Stability defined in the ideal two-body model: long-term J2, drag, and third bodies are neglected when defining stability.

## Next steps
- Swap placeholder atmosphere with real US76/NRLMSIS data or tables.  
- Add events (burnout, stage separation) and a guidance parameterization to optimize over.  
- Add checks for dynamic pressure and axial acceleration limits, plus basic output plots.

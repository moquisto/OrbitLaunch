"""
Simulation glue: guidance, events, logging, and integration loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

import os
import numpy as np

from Environment.aerodynamics import Aerodynamics, get_wind_at_altitude
from Environment.atmosphere import AtmosphereModel
from Environment.gravity import EarthModel, orbital_elements_from_state
from Hardware.rocket import Rocket
from Software.guidance import Guidance
from Software.events import EventManager # Import EventManager
from .integrators import Integrator, RK4
from .state import State
from .telemetry import Logger
from .config import SimulationConfig
from Environment.config import EnvironmentConfig
from Logging.config import LoggingConfig


@dataclass
class ControlCommand:
    throttle: float
    thrust_direction_eci: np.ndarray  # unit vector in ECI


class Simulation:
    def __init__(
        self,
        earth: EarthModel,
        atmosphere: AtmosphereModel,
        aerodynamics: Aerodynamics,
        rocket: Rocket,
        sim_config: SimulationConfig,
        env_config: EnvironmentConfig,
        log_config: LoggingConfig,
        integrator: Optional[Integrator] = None,
        guidance: Optional[Guidance] = None,
        sw_config: Optional[Any] = None, # Add sw_config
        event_manager: Optional[EventManager] = None, # Add EventManager
    ):
        self.earth = earth
        self.atmosphere = atmosphere
        self.aero = aerodynamics
        self.rocket = rocket
        self.sim_config = sim_config
        self.env_config = env_config
        self.log_config = log_config
        self.sw_config = sw_config # Store sw_config
        self.integrator = integrator or RK4()
        self.guidance = guidance or Guidance(sw_config=sw_config, env_config=env_config, # Provide default Guidance params if none given
                                            pitch_program=None, upper_throttle_program=None,
                                            booster_throttle_schedule=[], rocket_stages_info=[])
        self.event_manager = event_manager or EventManager(rocket=rocket) # Store EventManager
        
        # Store relevant config values to avoid passing the whole object around
        self.max_q_limit = sim_config.max_q_limit
        self.max_accel_limit = sim_config.max_accel_limit
        self.impact_altitude_buffer_m = log_config.impact_altitude_buffer_m
        self.escape_radius_factor = log_config.escape_radius_factor
        self.use_jet_stream_model = env_config.use_jet_stream_model
        self.air_gamma = env_config.air_gamma
        self.air_gas_constant = env_config.air_gas_constant

        # Tiny memoization cache for atmospheric properties (cleared each run)
        self._atmo_cache: dict[tuple[float, float], Any] = {}

    def _current_stage_propellant_mass(self, state: State) -> float:
        """Estimate active-stage propellant remaining from total mass.

        State.m tracks total vehicle mass. Since this model drops whole stages at
        separation, the total mass at any instant equals:
          dry(active) + prop(active, remaining) + sum(total_mass(later stages))
        """
        try:
            idx = int(getattr(state, "stage_index", 0))
        except Exception:
            idx = 0
        idx = int(np.clip(idx, 0, len(self.rocket.stages) - 1))
        mass_total = float(getattr(state, "m", 0.0) or 0.0)
        dry_active = float(getattr(self.rocket.stages[idx], "dry_mass", 0.0) or 0.0)
        later_total = 0.0
        for j in range(idx + 1, len(self.rocket.stages)):
            try:
                later_total += float(self.rocket.stages[j].total_mass())
            except Exception:
                later = self.rocket.stages[j]
                later_total += float(getattr(later, "dry_mass", 0.0) or 0.0) + float(
                    getattr(later, "prop_mass", 0.0) or 0.0
                )
        return max(0.0, mass_total - dry_active - later_total)

    def _rhs(self, t_env: float, t_sim: float, state: State, control: ControlCommand):
        r = np.asarray(state.r_eci, dtype=float)
        v = np.asarray(state.v_eci, dtype=float)
        r_norm = np.linalg.norm(r)
        v_norm = np.linalg.norm(v)
        mass = max(float(state.m), 1e-6)  # prevent divide-by-zero

        # Gravity
        a_grav = self.earth.gravity_accel(r)

        # Atmosphere and drag
        altitude = max(0.0, r_norm - float(self.earth.radius))
        props_key = (altitude, t_env)
        if props_key in self._atmo_cache:
            props = self._atmo_cache[props_key]
        else:
            props = self.atmosphere.properties(altitude, t_env)
            self._atmo_cache[props_key] = props
        p_amb = float(props.p)

        F_drag = self.aero.drag_force(state, self.earth, t_env, self.rocket)
        
        # Calculate dynamic pressure 'q' and Mach number.
        rho = float(props.rho)
        v_atm_rotation = self.earth.atmosphere_velocity(r)
        wind_vector = get_wind_at_altitude(altitude, self.env_config, r) if self.env_config.use_jet_stream_model else np.zeros(3)
        v_air = v_atm_rotation + wind_vector
        v_rel = v - v_air
        v_rel_mag = np.linalg.norm(v_rel)
        q = 0.5 * rho * v_rel_mag**2 if rho > 0 else 0.0

        a_sound = np.sqrt(max(self.air_gamma * self.air_gas_constant * float(props.T), 0.0))
        mach = v_rel_mag / a_sound if a_sound > 0.0 else 0.0

        throttle = control.throttle
        if self.max_q_limit is not None and q > self.max_q_limit and q > 0.0:
            throttle = float(np.clip(throttle * (self.max_q_limit / q), 0.0, 1.0))

        # Thrust + mass flow (using refactored rocket method)
        current_prop_mass = self._current_stage_propellant_mass(state)
        F_thrust, dm_dt = self.rocket.thrust_and_mass_flow(
            t_sim, # Pass current simulation time
            throttle,
            control.thrust_direction_eci,
            state,
            p_amb,
            current_prop_mass,
        )

        # Accelerations
        a_drag = F_drag / mass
        a_thrust = F_thrust / mass
        accel_mag = np.linalg.norm(a_grav + a_drag + a_thrust)
        if self.max_accel_limit is not None and accel_mag > self.max_accel_limit and accel_mag > 0.0:
            scale = self.max_accel_limit / accel_mag
            F_thrust *= scale
            a_thrust = F_thrust / mass
            dm_dt *= scale

        dr_dt = v
        dv_dt = a_grav + a_drag + a_thrust
        dm_dt = float(dm_dt)

        # Diagnostics
        v_rel_mag = np.linalg.norm(v_rel) if 'v_rel' in locals() else 0.0
        a_sound = np.sqrt(max(self.air_gamma * self.air_gas_constant * float(props.T), 0.0))
        mach = v_rel_mag / a_sound if a_sound > 0.0 else 0.0
        vr = float(np.dot(v, r / r_norm)) if r_norm > 0 else 0.0
        v_horiz = float(np.sqrt(max(v_norm**2 - vr**2, 0.0)))
        fpa_deg = float(np.degrees(np.arctan2(vr, v_horiz))) if (v_horiz > 0 or vr != 0) else 0.0
        specific_energy = 0.5 * v_norm**2 - self.earth.mu / r_norm if r_norm > 0 else 0.0

        extras = {
            "altitude": altitude,
            "speed": v_norm,
            "thrust_mag": np.linalg.norm(F_thrust),
            "drag_mag": np.linalg.norm(F_drag),
            "mdot": dm_dt,
            "dynamic_pressure": q,
            "rho": rho,
            "mach": mach,
            "fpa_deg": fpa_deg,
            "v_vertical": vr,
            "v_horizontal": v_horiz,
            "specific_energy": specific_energy,
        }

        return dr_dt, dv_dt, dm_dt, extras

    def run(
        self,
        t_env_start: float,
        duration: float,
        dt: float,
        state0: State,
    ) -> Logger:
        """
        March the simulation forward from t0 to tf with fixed step dt.
        """
        logger = Logger()
        t_sim = 0.0
        t_env = float(t_env_start)
        t_end_sim = duration
        state = state0
    
        # Reset stateful components for a fresh run.
        if hasattr(self.rocket, "reset"):
            self.rocket.reset()
        self.guidance.reset()
        self._atmo_cache.clear()
        
        # Ensure the simulation starts with the correct stage index from the initial state.
        state.stage_index = int(getattr(state, "stage_index", 0))

        orbit_coast_end: float | None = None
        orbit_cutoff_time: float | None = None

        direct_cutoff_enabled = str(os.getenv("ORBITLAUNCH_DIRECT_INSERTION_CUTOFF", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            direct_cutoff_tol_m = float(os.getenv("ORBITLAUNCH_DIRECT_INSERTION_TOL_M", "10000"))
        except Exception:
            direct_cutoff_tol_m = 10_000.0
        direct_cutoff_tol_m = max(0.0, float(direct_cutoff_tol_m))

        target_alt_m = self.sim_config.target_orbit_alt_m
        target_r_m = None
        if target_alt_m is not None and np.isfinite(float(target_alt_m)):
            target_r_m = float(self.earth.radius) + float(target_alt_m)

        while t_sim <= t_end_sim:
            # --- Main Simulation Loop ---
            # Cache derivatives/extras so we don't recompute for logging + integrator.
            rhs_cache: dict[tuple[float, int], tuple[np.ndarray, np.ndarray, float, State, dict]] = {} # Update key type to include State

            def rhs_cached(tau: float, s: State):
                nonlocal orbit_coast_end, orbit_cutoff_time

                key = (tau, id(s))
                if key not in rhs_cache:
                    # Guidance needs Mach at the *substep* state/time. Using a
                    # fixed Mach computed at the start of the outer step makes
                    # MECO timing dt-dependent and can shift the achieved orbit
                    # by tens of km.
                    t_env_tau = t_env + (tau - t_sim)
                    r_tau = np.asarray(getattr(s, "r_eci", np.zeros(3)), dtype=float)
                    v_tau = np.asarray(getattr(s, "v_eci", np.zeros(3)), dtype=float)
                    r_norm_tau = float(np.linalg.norm(r_tau))
                    altitude_tau = max(0.0, r_norm_tau - float(self.earth.radius))
                    props_key = (altitude_tau, float(t_env_tau))
                    if props_key in self._atmo_cache:
                        props_tau = self._atmo_cache[props_key]
                    else:
                        props_tau = self.atmosphere.properties(altitude_tau, float(t_env_tau))
                        self._atmo_cache[props_key] = props_tau

                    v_atm_rotation_tau = self.earth.atmosphere_velocity(r_tau)
                    wind_vector_tau = (
                        get_wind_at_altitude(altitude_tau, self.env_config, r_tau)
                        if self.use_jet_stream_model
                        else np.zeros(3)
                    )
                    v_air_tau = v_atm_rotation_tau + wind_vector_tau
                    v_rel_tau = v_tau - v_air_tau
                    v_rel_mag_tau = float(np.linalg.norm(v_rel_tau))
                    a_sound_tau = float(
                        np.sqrt(max(self.air_gamma * self.air_gas_constant * float(props_tau.T), 0.0))
                    )
                    mach_tau = float(v_rel_mag_tau / a_sound_tau) if a_sound_tau > 0.0 else 0.0

                    # Guidance needs current propellant mass and mach
                    current_prop_mass_for_guidance = self._current_stage_propellant_mass(s)
                    guidance_command = self.guidance.compute_command(
                        tau,
                        s,
                        current_prop_mass_for_guidance,
                        mach_tau,
                    )
                    
                    # Apply events based on guidance command BEFORE integration step
                    # This 's' is a copy, so modifying it here will be local to this rhs_cached call
                    s_after_events = self.event_manager.apply_events(s.copy(), guidance_command) # Apply to a copy to avoid unexpected side-effects within deriv_fn calls

                    # Optional: direct-insertion cutoff. If the achieved bound orbit is
                    # within tolerance on *both* perigee and apoapsis, command engine
                    # shutdown from this substep onward (no post-circ burn needed).
                    if (
                        direct_cutoff_enabled
                        and orbit_cutoff_time is None
                        and target_r_m is not None
                        and int(getattr(s_after_events, "stage_index", 0)) == 1
                        and float(getattr(guidance_command, "throttle", 0.0) or 0.0) > 1e-3
                    ):
                        a_m, rp_m, ra_m = orbital_elements_from_state(r_tau, v_tau, float(self.earth.mu))
                        if (
                            a_m is not None
                            and rp_m is not None
                            and ra_m is not None
                            and np.isfinite(float(a_m))
                            and float(a_m) > 0.0
                            and np.isfinite(float(rp_m))
                            and np.isfinite(float(ra_m))
                            and abs(float(rp_m) - float(target_r_m)) <= direct_cutoff_tol_m
                            and abs(float(ra_m) - float(target_r_m)) <= direct_cutoff_tol_m
                        ):
                            orbit_cutoff_time = float(tau)
                            logger.orbit_achieved = True
                            logger.cutoff_reason = "orbit_target_met"
                            if self.sim_config.exit_on_orbit:
                                if float(self.sim_config.post_orbit_coast_s) > 0.0:
                                    orbit_coast_end = float(tau) + float(self.sim_config.post_orbit_coast_s)
                                else:
                                    orbit_coast_end = float(tau)

                    if orbit_cutoff_time is not None and tau >= orbit_cutoff_time:
                        guidance_command.throttle = 0.0

                    # Pass the guidance command as control to _rhs
                    dr_dt, dv_dt, dm_dt, _extras = self._rhs(t_env_tau, tau, s_after_events, guidance_command)
                    rhs_cache[key] = (dr_dt, dv_dt, dm_dt, s_after_events, _extras) # Store updated state as well
                dr_dt, dv_dt, dm_dt, updated_s, _extras = rhs_cache[key]
                return dr_dt, dv_dt, dm_dt, updated_s # Return updated state

            # Trigger first evaluation (k1) for current state/time and log using it.
            # rhs_cached now returns (dr_dt, dv_dt, dm_dt, updated_s)
            drdt, dvdt, dmdt, current_s_after_events = rhs_cached(t_sim, state) # Unpack all 4 values

            # Extract extras from the cache
            # The key for rhs_cache is (tau, id(s)), and the value in cache is (dr_dt, dv_dt, dm_dt, s_after_events, _extras)
            # So _extras is at index 4.
            extras = rhs_cache[(t_sim, id(state))][4] # Correctly access extras at index 4

            logger.record(t_sim, t_env, current_s_after_events, extras) # Use the updated state for logging

            # --- TERMINATION & EVENT CHECKS ---

            # 1. ORBIT ACHIEVED (local-state check; used for early termination)
            if self.sim_config.target_orbit_alt_m is not None and not logger.orbit_achieved:
                r_norm = np.linalg.norm(state.r_eci)
                v_norm = np.linalg.norm(state.v_eci)
                r_hat = state.r_eci / r_norm
                vr = float(np.dot(state.v_eci, r_hat))
                target_r = float(self.earth.radius) + float(self.sim_config.target_orbit_alt_m)
                v_circ = np.sqrt(self.earth.mu / target_r)
                if (
                    abs(r_norm - target_r) <= self.sim_config.orbit_alt_tol
                    and abs(v_norm - v_circ) <= self.sim_config.orbit_speed_tol
                    and abs(vr) <= self.sim_config.orbit_radial_tol
                ):
                    logger.orbit_achieved = True
                    logger.cutoff_reason = "orbit_target_met"
                    if orbit_cutoff_time is None:
                        orbit_cutoff_time = float(t_sim)

                    if self.sim_config.exit_on_orbit:
                        if self.sim_config.post_orbit_coast_s > 0.0:
                            orbit_coast_end = float(t_sim) + float(self.sim_config.post_orbit_coast_s)
                        else:
                            break

            if orbit_coast_end is not None and t_sim >= orbit_coast_end:
                if not logger.cutoff_reason:
                    logger.cutoff_reason = "coast_complete"
                break

            # 2. EARLY TERMINATION (FAILURE CONDITIONS)
            r_norm = np.linalg.norm(state.r_eci)
            altitude = r_norm - self.earth.radius
            if altitude < self.impact_altitude_buffer_m:
                logger.cutoff_reason = "impact"
                break
            
            specific_energy = 0.5 * np.dot(state.v_eci, state.v_eci) - self.earth.mu / r_norm
            vr = float(np.dot(state.v_eci, state.r_eci / r_norm)) if r_norm > 0 else 0.0
            if specific_energy > 0 and vr > 0 and r_norm > self.escape_radius_factor * self.earth.radius:
                logger.cutoff_reason = "escape"
                break

            # --- INTEGRATION & GUIDANCE-DRIVEN EVENTS ---
            # Events are now applied inside rhs_cached via EventManager.apply_events
            # This ensures state updates from events are incorporated before integration step.
            
            # Integrate the state
            state = self.integrator.step(rhs_cached, state, t_sim, dt)

            t_sim += dt
            t_env += dt

        return logger

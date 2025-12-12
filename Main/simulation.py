"""
Simulation glue: guidance, events, logging, and integration loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

import numpy as np

from Environment.aerodynamics import Aerodynamics, get_wind_at_altitude
from Environment.atmosphere import AtmosphereModel
from Environment.gravity import EarthModel
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

        # Internal state for propellant tracking
        self._propellant_remaining_current_stage: float = self.rocket.stages[0].prop_mass
        self._total_dry_mass_remaining: float = sum(s.dry_mass for s in self.rocket.stages)

        # Tiny memoization cache for atmospheric properties (cleared each run)
        self._atmo_cache: dict[tuple[float, float], Any] = {}

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
        current_prop_mass = self._propellant_remaining_current_stage
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

        # Initial propellant state for simulation (based on initial stage)
        self._propellant_remaining_current_stage = self.rocket.stages[state.stage_index].prop_mass
        self._total_dry_mass_remaining = sum(s.dry_mass for s in self.rocket.stages)

        orbit_coast_end: float | None = None

        while t_sim <= t_end_sim:
            # --- Main Simulation Loop ---

            # Pre-calculate atmospheric properties and Mach for Guidance
            r_norm_current = np.linalg.norm(state.r_eci)
            altitude_current = max(0.0, r_norm_current - float(self.earth.radius))
            
            props_current = self.atmosphere.properties(altitude_current, t_env)
            rho_current = float(props_current.rho)

            v_atm_rotation_current = self.earth.atmosphere_velocity(state.r_eci)
            wind_vector_current = self.aero._get_wind_at_altitude(altitude_current, state.r_eci) if self.use_jet_stream_model else np.zeros(3)
            v_air_current = v_atm_rotation_current + wind_vector_current
            v_rel_current = np.asarray(state.v_eci, dtype=float) - v_air_current
            v_rel_mag_current = np.linalg.norm(v_rel_current)
            a_sound_current = np.sqrt(max(self.air_gamma * self.air_gas_constant * float(props_current.T), 0.0))
            mach_current = v_rel_mag_current / a_sound_current if a_sound_current > 0.0 else 0.0


            # Cache derivatives/extras so we don't recompute for logging + integrator.
            rhs_cache: dict[tuple[float, int], tuple[np.ndarray, np.ndarray, float, dict]] = {}

            def rhs_cached(tau: float, s: State):
                key = (tau, id(s))
                if key not in rhs_cache:
                    # Guidance needs current propellant mass and mach
                    current_prop_mass_for_guidance = self._propellant_remaining_current_stage
                    guidance_command = self.guidance.compute_command(tau, s, current_prop_mass_for_guidance, mach_current)
                    
                    # Apply events based on guidance command BEFORE integration step
                    s = self.event_manager.apply_events(s, guidance_command)

                    t_env_tau = t_env + (tau - t_sim)
                    # Pass the guidance command as control to _rhs
                    rhs_cache[key] = self._rhs(t_env_tau, tau, s, guidance_command)
                dr_dt, dv_dt, dm_dt, _extras = rhs_cache[key]
                return dr_dt, dv_dt, dm_dt

            # Trigger first evaluation (k1) for current state/time and log using it.
            drdt, dvdt, dmdt = rhs_cached(t_sim, state)
            extras = rhs_cache[(t_sim, id(state))][3]
            logger.record(t_sim, t_env, state, extras)

            # Update propellant remaining for the current stage in Simulation's internal state
            if dmdt < 0: # dm_dt is negative for mass loss
                self._propellant_remaining_current_stage = max(0.0, self._propellant_remaining_current_stage + dmdt * dt)

            # --- TERMINATION & EVENT CHECKS ---

            # 1. ORBIT ACHIEVED
            if self.sim_config.target_orbit_alt_m is not None and orbit_coast_end is None:
                r_norm = np.linalg.norm(state.r_eci)
                v_norm = np.linalg.norm(state.v_eci)
                r_hat = state.r_eci / r_norm
                vr = float(np.dot(state.v_eci, r_hat))
                v_circ = np.sqrt(self.earth.mu / self.sim_config.target_orbit_alt_m)
                if (
                    abs(r_norm - self.sim_config.target_orbit_alt_m) <= self.sim_config.orbit_alt_tol
                    and abs(v_norm - v_circ) <= self.sim_config.orbit_speed_tol
                    and abs(vr) <= self.sim_config.orbit_radial_tol
                ):
                    logger.orbit_achieved = True
                    logger.cutoff_reason = "orbit_target_met"
                    if self.sim_config.exit_on_orbit:
                        if self.sim_config.post_orbit_coast_s > 0.0:
                            orbit_coast_end = t_sim + self.sim_config.post_orbit_coast_s
                        else:
                            break
                    else:
                        orbit_coast_end = t_end_sim if self.sim_config.post_orbit_coast_s <= 0.0 else t_sim + self.sim_config.post_orbit_coast_s

            if orbit_coast_end is not None and t_sim >= orbit_coast_end:
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


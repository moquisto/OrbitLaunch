"""
Simulation glue: guidance, events, logging, and integration loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

import numpy as np

from aerodynamics import Aerodynamics, get_wind_at_altitude
from atmosphere import AtmosphereModel
from gravity import EarthModel
from integrators import Integrator, RK4, State
from rocket import Rocket
from config import CFG


@dataclass
class ControlCommand:
    throttle: float
    thrust_direction: np.ndarray  # unit vector in ECI


class Guidance:
    """
    Deterministic guidance: provides throttle and thrust direction.
    If no pitch/throttle functions are supplied, defaults to prograde thrust
    and full throttle.
    """

    def __init__(
        self,
        pitch_program: Optional[Callable[[float, State], np.ndarray]] = None,
        throttle_schedule: Optional[Callable[[float, State], float]] = None,
    ):
        self.pitch_program = pitch_program
        self.throttle_schedule = throttle_schedule
        self.reset()

    def reset(self):
        """Reset the internal state of the guidance logic for a new run."""
        self._stage_start_time = 0.0
        self._last_stage_index = None

    def compute_command(self, t: float, state: State) -> ControlCommand:
        stage_idx = int(getattr(state, "stage_index", 0))
        if self._last_stage_index is None:
            self._stage_start_time = t
        elif stage_idx != self._last_stage_index:
            self._stage_start_time = t
        self._last_stage_index = stage_idx
        t_stage = t - self._stage_start_time

        # Thrust direction: user pitch program or prograde / +z fallback.
        if self.pitch_program is not None:
            try:
                direction = np.asarray(
                    self.pitch_program(t, state, t_stage=t_stage, stage_index=stage_idx),
                    dtype=float,
                )
            except TypeError:
                direction = np.asarray(self.pitch_program(t, state), dtype=float)
        else:
            v = np.asarray(state.v_eci, dtype=float)
            speed = np.linalg.norm(v)
            if speed > 0.0:
                direction = v / speed
            else:
                # If stationary, push along local up (radial); fall back to +z only if r is zero.
                r = np.asarray(state.r_eci, dtype=float)
                r_norm = np.linalg.norm(r)
                if r_norm > 0.0:
                    direction = r / r_norm
                else:
                    direction = np.array([0.0, 0.0, 1.0], dtype=float)

        norm = np.linalg.norm(direction)
        if norm == 0.0:
            direction = np.array([0.0, 0.0, 1.0], dtype=float)
            norm = 1.0
        direction_unit = direction / norm

        # Throttle schedule: user-supplied or full throttle.
        throttle = 1.0 if self.throttle_schedule is None else float(self.throttle_schedule(t, state))
        throttle = float(np.clip(throttle, 0.0, 1.0))

        return ControlCommand(throttle=throttle, thrust_direction=direction_unit)


class Logger:
    """
    Minimal in-memory logger for trajectories.
    """

    def __init__(self):
        self.t_sim = []
        self.t_env = []
        self.r = []
        self.v = []
        self.m = []
        self.stage = []
        self.altitude = []
        self.speed = []
        self.thrust_mag = []
        self.drag_mag = []
        self.mdot = []
        self.dynamic_pressure = []
        self.rho = []
        self.mach = []
        self.flight_path_angle_deg = []
        self.v_vertical = []
        self.v_horizontal = []
        self.specific_energy = []
        self.orbit_achieved = False
        self.cutoff_reason = ""

    def record(self, t_sim: float, t_env: float, state: State, extras: Dict[str, Any]):
        self.t_sim.append(float(t_sim))
        self.t_env.append(float(t_env))
        self.r.append(np.asarray(state.r_eci, dtype=float).copy())
        self.v.append(np.asarray(state.v_eci, dtype=float).copy())
        self.m.append(float(state.m))
        self.stage.append(int(getattr(state, "stage_index", 0)))
        self.altitude.append(float(extras.get("altitude", 0.0)))
        self.speed.append(float(extras.get("speed", 0.0)))
        self.thrust_mag.append(float(extras.get("thrust_mag", 0.0)))
        self.drag_mag.append(float(extras.get("drag_mag", 0.0)))
        self.mdot.append(float(extras.get("mdot", 0.0)))
        self.dynamic_pressure.append(float(extras.get("dynamic_pressure", 0.0)))
        self.rho.append(float(extras.get("rho", 0.0)))
        self.mach.append(float(extras.get("mach", 0.0)))
        self.flight_path_angle_deg.append(float(extras.get("fpa_deg", 0.0)))
        self.v_vertical.append(float(extras.get("v_vertical", 0.0)))
        self.v_horizontal.append(float(extras.get("v_horizontal", 0.0)))
        self.specific_energy.append(float(extras.get("specific_energy", 0.0)))


class Simulation:
    def __init__(
        self,
        earth: EarthModel,
        atmosphere: AtmosphereModel,
        aerodynamics: Aerodynamics,
        rocket: Rocket,
        integrator: Optional[Integrator] = None,
        guidance: Optional[Guidance] = None,
        max_q_limit: float | None = None,
        max_accel_limit: float | None = None,
        impact_altitude_buffer_m: float = -100.0,
        escape_radius_factor: float = 1.05,
    ):
        self.earth = earth
        self.atmosphere = atmosphere
        self.aero = aerodynamics
        self.rocket = rocket
        self.integrator = integrator or RK4()
        self.guidance = guidance or Guidance()
        self.max_q_limit = max_q_limit
        self.max_accel_limit = max_accel_limit
        self.impact_altitude_buffer_m = impact_altitude_buffer_m
        self.escape_radius_factor = escape_radius_factor
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
        
        # Calculate dynamic pressure 'q' for max-q throttle limiting.
        # This requires air-relative velocity, which is also calculated inside drag_force.
        # To avoid changing the drag_force signature, we recalculate it here.
        # This is a minor inefficiency but keeps the component interface clean.
        rho = float(props.rho)
        if rho > 0:
            v_atm_rotation = self.earth.atmosphere_velocity(r)
            wind_vector = get_wind_at_altitude(altitude, r) if CFG.use_jet_stream_model else np.zeros(3)
            v_air = v_atm_rotation + wind_vector
            v_rel = v - v_air
            q = 0.5 * rho * np.dot(v_rel, v_rel)
        else:
            q = 0.0

        throttle = control.throttle
        if self.max_q_limit is not None and q > self.max_q_limit and q > 0.0:
            throttle = float(np.clip(throttle * (self.max_q_limit / q), 0.0, 1.0))

        # Thrust + mass flow
        control_payload = {
            "t": t_sim,
            "throttle": throttle,
            "thrust_dir_eci": control.thrust_direction,
            "dir_is_unit": True,
        }
        F_thrust, dm_dt = self.rocket.thrust_and_mass_flow(control_payload, state, p_amb)

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
        a_sound = np.sqrt(max(CFG.air_gamma * CFG.air_gas_constant * float(props.T), 0.0))
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
        orbit_target_radius: float | None = None,
        orbit_speed_tolerance: float = 50.0,
        orbit_radial_tolerance: float = 50.0,
        orbit_alt_tolerance: float = 500.0,
        exit_on_orbit: bool = True,
        post_orbit_coast_s: float = 0.0,
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
        self.rocket.reset()
        self.guidance.reset()
        self._atmo_cache.clear()
        
        # Ensure the simulation starts with the correct stage index from the initial state.
        state.stage_index = int(getattr(state, "stage_index", 0))
        
        orbit_coast_end: float | None = None

        while t_sim <= t_end_sim:
            # --- Main Simulation Loop ---

            # DEFENSIVE GUARD: Prevent a stale stage_index=1 if the booster still has fuel
            # and no shutdown/separation timeline has been established.
            if (
                getattr(state, "stage_index", 0) > 0
                and self.rocket.stage_fuel_empty_time[0] is None
                and self.rocket.stage_prop_remaining[0] > 0
                and self.rocket.stage_engine_off_complete_time[0] is None
                and self.rocket.separation_time_planned is None
            ):
                state.stage_index = 0

            # Cache derivatives/extras so we don't recompute for logging + integrator.
            rhs_cache: dict[tuple[float, int], tuple[np.ndarray, np.ndarray, float, dict]] = {}

            def rhs_cached(tau: float, s: State):
                key = (tau, id(s))
                if key not in rhs_cache:
                    control_local = self.guidance.compute_command(tau, s)
                    t_env_tau = t_env + (tau - t_sim)
                    rhs_cache[key] = self._rhs(t_env_tau, tau, s, control_local)
                dr_dt, dv_dt, dm_dt, _extras = rhs_cache[key]
                return dr_dt, dv_dt, dm_dt

            # Trigger first evaluation (k1) for current state/time and log using it.
            drdt, dvdt, dmdt = rhs_cached(t_sim, state)
            extras = rhs_cache[(t_sim, id(state))][3]
            logger.record(t_sim, t_env, state, extras)

            # --- TERMINATION & EVENT CHECKS ---

            # 1. ORBIT ACHIEVED
            if orbit_target_radius is not None and orbit_coast_end is None:
                r_norm = np.linalg.norm(state.r_eci)
                v_norm = np.linalg.norm(state.v_eci)
                r_hat = state.r_eci / r_norm
                vr = float(np.dot(state.v_eci, r_hat))
                v_circ = np.sqrt(self.earth.mu / orbit_target_radius)
                if (
                    abs(r_norm - orbit_target_radius) <= orbit_alt_tolerance
                    and abs(v_norm - v_circ) <= orbit_speed_tolerance
                    and abs(vr) <= orbit_radial_tolerance
                ):
                    logger.orbit_achieved = True
                    logger.cutoff_reason = "orbit_target_met"
                    if exit_on_orbit:
                        if post_orbit_coast_s > 0.0:
                            orbit_coast_end = t_sim + post_orbit_coast_s
                        else:
                            break
                    else:
                        orbit_coast_end = t_end_sim if post_orbit_coast_s <= 0.0 else t_sim + post_orbit_coast_s

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

            # --- INTEGRATION & STAGING ---
            state = self.integrator.step(rhs_cached, state, t_sim, dt)

            # 3. STAGE SEPARATION
            if (
                self.rocket.separation_time_planned is not None
                and t_sim >= self.rocket.separation_time_planned
                and getattr(state, "stage_index", 0) == 0
            ):
                stage_idx = 0
                remaining_prop = self.rocket.stage_prop_remaining[stage_idx]
                dropped_mass = self.rocket.stages[stage_idx].dry_mass + remaining_prop
                state.m = max(state.m - dropped_mass, 1e-6)
                self.rocket.stage_prop_remaining[stage_idx] = 0.0
                state.stage_index = 1
                if self.rocket.upper_ignition_start_time is None:
                    self.rocket.upper_ignition_start_time = t_sim + self.rocket.upper_ignition_delay
                
                # Propagate ignition time to state for other components to see
                setattr(state, 'upper_ignition_start_time', self.rocket.upper_ignition_start_time)
                
                self.rocket.separation_time_planned = None

            t_sim += dt
            t_env += dt

        return logger

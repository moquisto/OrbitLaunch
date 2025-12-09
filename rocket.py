
"""
Vehicle model: outlines for engines, stages, and rocket behavior.
Implementations are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from gravity import R_EARTH

# -------------------------------
# Physical constants
# -------------------------------
G0 = 9.80665  # [m/s^2] standard gravity for Isp conversion
P_SL = 101325.0  # [Pa] reference sea-level pressure
A0 = 340.0  # [m/s] reference speed of sound used for Mach estimate


@dataclass
class Engine:
    thrust_vac: float
    thrust_sl: float
    isp_vac: float
    isp_sl: float

    def thrust_and_isp(self, throttle: float, p_amb: float) -> Tuple[float, float]:
        """
        Compute engine thrust and Isp for a given throttle and ambient pressure.

        Parameters
        ----------
        throttle : float
            Commanded throttle in [0, 1].
        p_amb : float
            Ambient static pressure [Pa]. Typically provided by the atmosphere
            model using the current altitude.

        Returns
        -------
        thrust : float
            Thrust magnitude [N] at the requested operating point.
        isp : float
            Specific impulse [s] at the requested operating point.

        Notes
        -----
        We interpolate linearly between sea-level (p = P_SL) and vacuum
        (p = 0) performance based on p_amb, then scale by throttle.
        """
        throttle = float(np.clip(throttle, 0.0, 1.0))

        # Clamp ambient pressure to [0, P_SL] for interpolation
        p = float(np.clip(p_amb, 0.0, P_SL))
        # Fraction of "vacuum-ness": 0 at sea level, 1 in vacuum
        f_vac = 1.0 - p / P_SL

        thrust_nominal = self.thrust_sl + f_vac * (self.thrust_vac - self.thrust_sl)
        isp_nominal = self.isp_sl + f_vac * (self.isp_vac - self.isp_sl)

        thrust = throttle * thrust_nominal
        return thrust, isp_nominal


@dataclass
class Stage:
    dry_mass: float
    prop_mass: float
    engine: Engine
    ref_area: float

    def total_mass(self) -> float:
        """
        Return total stage mass (dry + propellant).

        The time evolution of the vehicle's total mass is handled via the
        State.m variable in the integrator; this value is mainly useful
        for constructing the initial mass budget or for stage-drop deltas.
        """
        return self.dry_mass + self.prop_mass



class Rocket:
    """
    Simple two-stage rocket model.

    Assumptions
    -----------
    * stages[0] is the booster (first stage).
    * stages[1] is the upper stage / payload stage.
    * The overall vehicle mass is carried in State.m and is updated by the
      integrator using the returned mass-flow rate dm_dt.
    * This class does NOT mutate the State object; it only reads from it.
      Stage index changes (state.stage_index) and mass drops at separation
      should be handled in the outer simulation code.

    Throttle schedule
    -----------------
    * Booster (stage_index == 0):
        - Thrust ramps linearly from 0 to full over `main_engine_ramp_time`
          seconds from liftoff (t=0).
        - After ramp-up, thrust is held constant until the vehicle reaches
          `meco_mach` (approximate Mach number based on inertial speed).
        - At MECO, thrust is cut to zero permanently and the internal
          `meco_time` is recorded.

    * Upper stage (stage_index == 1):
        - No thrust until `meco_time` has been recorded and a jettison delay
          of `separation_delay` seconds has elapsed.
        - Then the upper-stage engine ramps linearly from 0 to full thrust
          over `upper_engine_ramp_time` seconds.
        - After ramp-up, thrust is kept constant.

    The user still decides when to actually change `state.stage_index`
    (and adjust State.m for the dropped booster). A convenient strategy is:
        - Use rocket.meco_time to detect MECO.
        - At t = meco_time + separation_delay, drop the booster mass and set
          state.stage_index = 1.
    """

    def __init__(
        self,
        stages: List[Stage],
        main_engine_ramp_time: float = 1.0,
        upper_engine_ramp_time: float = 1.0,
        meco_mach: float = 6.0,
        separation_delay: float = 30.0,
        upper_ignition_delay: float = 30.0,
        separation_altitude_m: Optional[float] = None,
        earth_radius: float = R_EARTH,
        min_throttle: float = 0.0,
    ):
        if len(stages) < 2:
            raise ValueError("Rocket expects at least two stages (booster + upper stage).")

        self.stages = stages
        self.main_engine_ramp_time = float(main_engine_ramp_time)
        self.upper_engine_ramp_time = float(upper_engine_ramp_time)
        self.meco_mach = float(meco_mach)
        self.separation_delay = float(separation_delay)
        self.upper_ignition_delay = float(upper_ignition_delay)
        self.separation_altitude_m = separation_altitude_m
        self.earth_radius = float(earth_radius)
        self.min_throttle = float(np.clip(min_throttle, 0.0, 1.0))

        # Internal state for event timing
        self.meco_time: float | None = None  # time when Mach first exceeds meco_mach
        self._last_time: float = 0.0

        # Per-stage propellant bookkeeping for scheduling (independent of State.m)
        self.stage_prop_remaining = [s.prop_mass for s in stages]
        self.stage_fuel_empty_time: list[float | None] = [None] * len(stages)
        self.stage_engine_off_complete_time: list[float | None] = [None] * len(stages)

        # Planned timeline events driven by stage 1 fuel depletion unless overridden by altitude:
        #  - separation_time_planned: booster jettison time (if no altitude specified)
        #  - upper_ignition_start_time: upper-stage ramp-up start time
        self.separation_time_planned: float | None = None
        self.upper_ignition_start_time: float | None = None

    # ------------------------------------------------------------------
    # Helper methods for stage selection and geometry
    # ------------------------------------------------------------------
    def current_stage_index(self, state) -> int:
        """
        Return the clamped stage index from the simulation State.

        The State is expected to have an integer attribute `stage_index`
        (see integrators.State). Values outside [0, len(stages)-1] are
        clamped.
        """
        idx = getattr(state, "stage_index", 0)
        idx = int(np.clip(idx, 0, len(self.stages) - 1))
        return idx

    def current_stage(self, state) -> Stage:
        """Return the currently active stage object."""
        return self.stages[self.current_stage_index(state)]

    def reference_area(self, state) -> float:
        """Return the reference area [m^2] to be used for drag."""
        return self.current_stage(state).ref_area

    # ------------------------------------------------------------------
    # Thrust and mass flow
    # ------------------------------------------------------------------
    def thrust_and_mass_flow(self, control, state, p_amb: float) -> Tuple[np.ndarray, float]:
        """
        Compute thrust vector and mass flow rate for the current state.

        Parameters
        ----------
        control :
            An object or dict providing at least:
                * t : float
                    Current simulation time [s].
                * throttle : float in [0, 1] (optional, default=1)
                    High-level throttle scaling (e.g. for guidance).
                * thrust_dir_eci : np.ndarray shape (3,) (optional)
                    Desired thrust direction in ECI frame. If omitted,
                    thrust is aligned with the velocity vector, or +z if
                    the vehicle is stationary.
        state :
            The current State instance (r_eci, v_eci, m, stage_index).
        p_amb : float
            Ambient static pressure [Pa], typically from the atmosphere
            model at the vehicle altitude.

        Returns
        -------
        thrust_vec : np.ndarray
            Thrust vector in ECI coordinates [N].
        dm_dt : float
            Mass flow rate [kg/s]. This value is negative when thrust
            is positive (propellant is consumed).
        """
        # Extract control fields in a robust way (support dicts and simple objects)
        if isinstance(control, dict):
            t = float(control.get("t", 0.0))
            throttle_cmd = float(control.get("throttle", 1.0))
            thrust_dir = control.get("thrust_dir_eci", None)
        else:
            t = float(getattr(control, "t", 0.0))
            throttle_cmd = float(getattr(control, "throttle", 1.0))
            thrust_dir = getattr(control, "thrust_dir_eci", None)

        # Estimate local time step based on the last call (for propellant bookkeeping)
        prev_t = self._last_time
        dt_internal = t - prev_t if t >= prev_t else 0.0
        self._last_time = t

        # Determine an approximate Mach number based on inertial speed.
        v_vec = np.asarray(state.v_eci, dtype=float)
        speed = float(np.linalg.norm(v_vec))
        mach = speed / A0 if A0 > 0.0 else 0.0

        stage_idx = self.current_stage_index(state)
        stage = self.stages[stage_idx]

        # Shape function for throttle based on stage, fuel and the requested timeline:
        # Stage 1 (index 0):
        #   - 1 s linear ramp-up from t=0 to full thrust
        #   - full thrust until fuel exhausted
        #   - 1 s linear ramp-down to zero
        #   - coast for separation_delay, then stage separation allowed
        # Stage 2 (index 1):
        #   - ignite after upper_ignition_start_time (= separation_time + upper_ignition_delay)
        #   - 1 s linear ramp-up to commanded thrust
        #   - full thrust until fuel exhausted
        #   - 1 s linear ramp-down to zero
        shape = 0.0

        if stage_idx == 0:
            # Booster / first stage
            fuel_empty_time = self.stage_fuel_empty_time[0]
            off_time = self.stage_engine_off_complete_time[0]

            if fuel_empty_time is None:
                # Fuel not exhausted yet: ramp-up then full thrust.
                if t < 0.0:
                    shape = 0.0
                elif t < self.main_engine_ramp_time:
                    # Linear ramp 0 -> 1
                    shape = max(0.0, t / self.main_engine_ramp_time)
                else:
                    shape = 1.0
            else:
                # Fuel has been flagged empty: ramp down over 1 s, then off.
                if t < fuel_empty_time:
                    # Before we "realize" fuel is gone, keep previous behaviour:
                    if t < self.main_engine_ramp_time:
                        shape = max(0.0, t / self.main_engine_ramp_time)
                    else:
                        shape = 1.0
                elif t < fuel_empty_time + 1.0:
                    # Linear ramp-down 1 -> 0 over 1 s
                    shape = max(0.0, 1.0 - (t - fuel_empty_time) / 1.0)
                else:
                    shape = 0.0
                    # Record engine-off complete time and schedule separation/upper ignition if not done yet.
                    if off_time is None:
                        off_time = fuel_empty_time + 1.0
                        self.stage_engine_off_complete_time[0] = off_time
                    if self.separation_time_planned is None:
                        # Schedule separation and upper ignition based on configured delays
                        self.separation_time_planned = off_time + self.separation_delay
                        self.upper_ignition_start_time = self.separation_time_planned + self.upper_ignition_delay

        else:
            # Upper stage / payload stage
            fuel_empty_time = self.stage_fuel_empty_time[1]
            off_time = self.stage_engine_off_complete_time[1]
            ignition_start = self.upper_ignition_start_time

            if ignition_start is None:
                # We have not yet scheduled upper-stage ignition
                shape = 0.0
            else:
                t_rel = t - ignition_start
                if fuel_empty_time is None:
                    # Fuel available: ramp-up then full.
                    if t_rel <= 0.0:
                        shape = 0.0
                    elif t_rel < self.upper_engine_ramp_time:
                        # Linear ramp 0 -> 1
                        shape = max(0.0, t_rel / self.upper_engine_ramp_time)
                    else:
                        shape = 1.0
                else:
                    # Fuel flagged empty: ramp-down over 1 s, then off.
                    if t < fuel_empty_time:
                        # Still before we consider it empty.
                        if t_rel <= 0.0:
                            shape = 0.0
                        elif t_rel < self.upper_engine_ramp_time:
                            shape = max(0.0, t_rel / self.upper_engine_ramp_time)
                        else:
                            shape = 1.0
                    elif t < fuel_empty_time + 1.0:
                        shape = max(0.0, 1.0 - (t - fuel_empty_time) / 1.0)
                    else:
                        shape = 0.0
                        if off_time is None:
                            self.stage_engine_off_complete_time[1] = fuel_empty_time + 1.0

        # Combine stage shape with commanded throttle
        effective_throttle = np.clip(throttle_cmd * shape, 0.0, 1.0)
        # Enforce minimum throttle when the engine is up to speed (shape ~1) and commanded on.
        if shape >= 0.99 and effective_throttle > 0.0 and self.min_throttle > 0.0:
            effective_throttle = max(effective_throttle, self.min_throttle)

        # Engine performance (uses ambient pressure from the atmosphere model)
        thrust_mag, isp = stage.engine.thrust_and_isp(effective_throttle, p_amb)

        # If propellant for the active stage is gone, force thrust to zero and
        # ensure we mark fuel depletion (prevents burning past empty).
        if self.stage_prop_remaining[stage_idx] <= 0.0:
            thrust_mag = 0.0
            if self.stage_fuel_empty_time[stage_idx] is None:
                self.stage_fuel_empty_time[stage_idx] = t

        # Direction: use control thrust_dir_eci if provided, otherwise align with velocity
        if thrust_dir is None:
            if speed > 0.0:
                dir_vec = v_vec / speed
            else:
                dir_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            dir_vec = np.asarray(thrust_dir, dtype=float)
            n = np.linalg.norm(dir_vec)
            if n == 0.0:
                # Fallback if a zero vector is passed
                dir_vec = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                dir_vec = dir_vec / n

        thrust_vec = thrust_mag * dir_vec

        # Mass flow: thrust = Isp * g0 * |dm_dt|
        if thrust_mag > 0.0 and isp > 0.0:
            dm_dt = -thrust_mag / (isp * G0)
        else:
            dm_dt = 0.0

        # ---------------------------------
        # Update internal propellant tracking for the active stage
        # ---------------------------------
        if dt_internal > 0.0 and dm_dt < 0.0:
            burned = -dm_dt * dt_internal  # positive propellant consumed
            self.stage_prop_remaining[stage_idx] = max(
                0.0, self.stage_prop_remaining[stage_idx] - burned
            )
            if (
                self.stage_prop_remaining[stage_idx] <= 0.0
                and self.stage_fuel_empty_time[stage_idx] is None
            ):
                # Record the first time we detect that this stage's fuel is empty
                self.stage_fuel_empty_time[stage_idx] = t

        return thrust_vec, dm_dt

    # ------------------------------------------------------------------
    # Stage separation helper
    # ------------------------------------------------------------------
    def stage_separation(self, state) -> bool:
        """
        Return True if, according to this model, the booster should have
        been jettisoned.

        This does NOT modify the State. The simulation code can use this
        as a trigger to:
            * subtract the booster dry mass from State.m, and
            * increment State.stage_index.

        Separation trigger rules:
            * If separation_altitude_m was provided at init, trigger when
              altitude >= separation_altitude_m.
            * Otherwise trigger when the scheduled separation_time_planned
              has been reached (set after fuel depletion and ramp-down).
        """
        if self.current_stage_index(state) > 0:
            # Already on upper stage
            return False

        # Compute altitude if possible (state must have r_eci)
        altitude = None
        if hasattr(state, "r_eci"):
            r_vec = np.asarray(state.r_eci, dtype=float)
            r_norm = np.linalg.norm(r_vec)
            altitude = max(0.0, r_norm - self.earth_radius)

        # Altitude-based trigger if specified
        if self.separation_altitude_m is not None and altitude is not None:
            if altitude >= float(self.separation_altitude_m):
                return True

        # Time-based trigger if scheduled
        if self.separation_time_planned is not None:
            return self._last_time >= self.separation_time_planned

        return False


# ------------------------------------------------------------------
# Self-test harness: vertical ascent simulation
# ------------------------------------------------------------------
if __name__ == "__main__":
    """
    Simple self-test for the Rocket class.

    We simulate a 1D vertical ascent from the Earth's surface with:
      * Two stages (booster + upper stage).
      * Booster thrust ramp-up, MECO at Mach ~6, 2 s stage-separation delay.
      * Upper stage ramp-up and then constant thrust.

    The environment model is deliberately simple:
      * Point-mass Earth gravity.
      * No aerodynamic drag (we only care about the thrust schedule here).
      * Ambient pressure comes from AtmosphereModel to exercise the engine
        sea-level vs vacuum interpolation.

    The integrator is a simple explicit Euler scheme; this test is *not* meant
    to be a high-fidelity trajectory simulation, only to check the qualitative
    behaviour of the Rocket class.
    """
    import matplotlib.pyplot as plt
    from dataclasses import dataclass
    from types import SimpleNamespace

    from atmosphere import AtmosphereModel

    # -----------------------------
    # Local test State definition
    # -----------------------------
    @dataclass
    class TestState:
        r_eci: np.ndarray
        v_eci: np.ndarray
        m: float
        stage_index: int = 0

    # -----------------------------
    # Build a sample two-stage rocket
    # -----------------------------
    # Approximate parameters inspired by SpaceX Super Heavy (booster)
    # and Starship (upper stage). These are order-of-magnitude values
    # taken from public sources (DLR analysis, Wikipedia, etc.), not
    # exact flight specs.
    booster_engine = Engine(
        thrust_vac=7.35e7,   # N  (Super Heavy max vacuum thrust ~73.5 MN)
        thrust_sl=7.0e7,     # N  (slightly lower at sea level)
        isp_vac=347.0,       # s  (Super Heavy Raptor vac Isp ~347 s)
        isp_sl=327.0,        # s  (Super Heavy Raptor SL Isp ~327 s)
    )
    # Approximate combined thrust of Starship upper stage engines
    # (3 sea-level + 3 vacuum Raptors), using rounded values.
    upper_engine = Engine(
        thrust_vac=1.5e7,    # N  (~15 MN total in vacuum)
        thrust_sl=1.2e7,     # N  (~12 MN total at sea level)
        isp_vac=380.0,       # s  (vacuum-optimised Raptors)
        isp_sl=330.0,        # s  (sea-level Raptors)
    )

    booster_stage = Stage(
        dry_mass=2.7e5,      # kg  (~270 t dry mass, Super Heavy order-of-magnitude)
        prop_mass=3.4e6,     # kg  (~3400 t propellant)
        engine=booster_engine,
        ref_area=np.pi * (4.5 ** 2),  # m^2, ~9 m diameter
    )
    upper_stage = Stage(
        dry_mass=1.3e5,      # kg  (~130 t dry mass)
        prop_mass=1.2e6,     # kg  (~1200 t propellant)
        engine=upper_engine,
        ref_area=np.pi * (4.5 ** 2),
    )

    rocket = Rocket(
        stages=[booster_stage, upper_stage],
        main_engine_ramp_time=1.0,
        upper_engine_ramp_time=1.0,
        meco_mach=6.0,
        separation_delay=2.0,
    )

    # Initial mass: sum of both stages
    m0 = booster_stage.total_mass() + upper_stage.total_mass()

    # -----------------------------
    # Environment and simulation setup
    # -----------------------------
    # Simple point-mass Earth
    R_E = 6371000.0           # m
    MU_E = 3.986004418e14     # m^3/s^2

    atm = AtmosphereModel()

    # Initial state: on the surface at the equator, pointing along +z
    r0 = np.array([0.0, 0.0, R_E], dtype=float)
    v0 = np.zeros(3, dtype=float)
    state = TestState(r_eci=r0.copy(), v_eci=v0.copy(), m=m0, stage_index=0)

    # Time integration parameters
    dt = 0.1        # s
    t_end = 1000   # s

    times = []
    altitudes = []
    speeds = []
    machs = []
    thrusts = []
    dm_dts = []
    stages_idx = []

    # Track events
    separation_time = None
    upper_full_throttle_time = None

    t = 0.0
    while t <= t_end:
        r_vec = state.r_eci
        v_vec = state.v_eci
        r_norm = float(np.linalg.norm(r_vec))
        alt = max(0.0, r_norm - R_E)
        speed = float(np.linalg.norm(v_vec))
        mach = speed / A0 if A0 > 0.0 else 0.0

        # Ambient pressure from the atmosphere model
        props = atm.properties(alt, t)
        p_amb = float(props.p)

        # Thrust direction: purely radial (upward)
        r_hat = r_vec / r_norm
        control = SimpleNamespace(
            t=t,
            throttle=1.0,
            thrust_dir_eci=r_hat,
        )

        thrust_vec, dm_dt = rocket.thrust_and_mass_flow(control, state, p_amb)
        thrust_mag = float(np.linalg.norm(thrust_vec))

        # Simple dynamics: point-mass gravity + thrust, no drag
        if r_norm > 0.0:
            a_grav = -MU_E * r_vec / (r_norm ** 3)
        else:
            a_grav = np.zeros(3)

        a_thrust = thrust_vec / state.m
        acc = a_grav + a_thrust

        # Explicit Euler integration step
        state.r_eci = state.r_eci + state.v_eci * dt
        state.v_eci = state.v_eci + acc * dt
        # For this simple test we keep mass constant to avoid refuelling logic;
        # we still record dm_dt so you can inspect its sign and magnitude.
        # Uncomment the next line if you want to see the effect of mass loss:
        state.m = max(state.m + dm_dt * dt, 1.0)

        # Stage separation logic: check once when the model says it should happen
        if rocket.stage_separation(state) and state.stage_index == 0:
            separation_time = t if separation_time is None else separation_time
            # Drop booster dry mass from the vehicle
            state.m -= booster_stage.dry_mass
            state.stage_index = 1

        # If upper stage should be at full throttle (according to the model timeline),
        # note the first time we cross that moment.
        if rocket.upper_ignition_start_time is not None:
            ignition_start = rocket.upper_ignition_start_time
            full_time = ignition_start + rocket.upper_engine_ramp_time
            if upper_full_throttle_time is None and t >= full_time:
                upper_full_throttle_time = t

        # Store data
        times.append(t)
        altitudes.append(alt / 1000.0)  # km
        speeds.append(speed)
        machs.append(mach)
        thrusts.append(thrust_mag)
        dm_dts.append(dm_dt)
        stages_idx.append(state.stage_index)

        # Stop if we "hit the ground" (numerical issues)
        if alt < 0.0:
            break

        t += dt

    times = np.array(times)
    altitudes = np.array(altitudes)
    speeds = np.array(speeds)
    machs = np.array(machs)
    thrusts = np.array(thrusts)
    dm_dts = np.array(dm_dts)
    stages_idx = np.array(stages_idx)

    # -----------------------------
    # Numeric summary for ChatGPT
    # -----------------------------
    print("\n=== Rocket class self-test: vertical ascent ===")
    print(f"Initial total mass          : {m0:.3e} kg")
    print(f"Booster stage mass (dry+prop): {booster_stage.total_mass():.3e} kg")
    print(f"Upper stage mass (dry+prop) : {upper_stage.total_mass():.3e} kg")
    print(f"Time step dt                : {dt} s")
    print(f"Total simulated time        : {times[-1]:.1f} s")

    booster_fuel_empty_time = rocket.stage_fuel_empty_time[0]
    booster_off_time = rocket.stage_engine_off_complete_time[0]

    if booster_fuel_empty_time is not None:
        idx_fe = np.argmin(np.abs(times - booster_fuel_empty_time))
        print(f"\nBooster fuel empty time           : {booster_fuel_empty_time:.2f} s")
        print(f"  Altitude at fuel empty          : {altitudes[idx_fe]:.2f} km")
        print(f"  Speed at fuel empty             : {speeds[idx_fe]:.1f} m/s")
        print(f"  Mach at fuel empty              : {machs[idx_fe]:.2f}")
    else:
        print("\nWARNING: Booster fuel never exhausted in this simulation.")

    if booster_off_time is not None:
        idx_off = np.argmin(np.abs(times - booster_off_time))
        print(f"Booster engine-off complete time  : {booster_off_time:.2f} s")
        print(f"  Altitude at engine off          : {altitudes[idx_off]:.2f} km")
    else:
        print("WARNING: Booster engine-off time not reached in this simulation.")

    if separation_time is not None:
        idx_sep = np.argmin(np.abs(times - separation_time))
        print(f"\nStage separation time (model trigger) : {separation_time:.2f} s")
        print(f"  Altitude at separation    : {altitudes[idx_sep]:.2f} km")
        print(f"  Speed at separation       : {speeds[idx_sep]:.1f} m/s")
        print(f"  Mach at separation        : {machs[idx_sep]:.2f}")
    else:
        print("\nWARNING: Stage separation did not occur in this simulation.")

    if upper_full_throttle_time is not None:
        idx_full = np.argmin(np.abs(times - upper_full_throttle_time))
        print(f"\nUpper stage full-throttle time (approx) : {upper_full_throttle_time:.2f} s")
        print(f"  Altitude at full throttle : {altitudes[idx_full]:.2f} km")
        print(f"  Mach at full throttle     : {machs[idx_full]:.2f}")
    else:
        print("\nINFO: Upper stage did not reach full throttle within the simulated time.")

    upper_fuel_empty_time = rocket.stage_fuel_empty_time[1]
    upper_off_time = rocket.stage_engine_off_complete_time[1]

    if upper_fuel_empty_time is not None:
        idx_ufe = np.argmin(np.abs(times - upper_fuel_empty_time))
        print(f"\nUpper stage fuel empty time       : {upper_fuel_empty_time:.2f} s")
        print(f"  Altitude at upper fuel empty    : {altitudes[idx_ufe]:.2f} km")
        print(f"  Mach at upper fuel empty        : {machs[idx_ufe]:.2f}")
    else:
        print("\nINFO: Upper stage fuel did not run out within the simulated time.")

    if upper_off_time is not None:
        idx_uoff = np.argmin(np.abs(times - upper_off_time))
        print(f"Upper stage engine-off time       : {upper_off_time:.2f} s")
        print(f"  Altitude at upper engine off    : {altitudes[idx_uoff]:.2f}")

    # Print a small table of samples every ~20 seconds
    print("\nSample timeline (every ~20 s):")
    print("  t [s]   stage   Mach    thrust [MN]   dm_dt [kg/s]   altitude [km]")
    for t_sample in np.arange(0.0, times[-1] + 1e-6, 20.0):
        idx = np.argmin(np.abs(times - t_sample))
        print(
            f"  {times[idx]:6.1f}   {stages_idx[idx]:5d}   "
            f"{machs[idx]:5.2f}   {thrusts[idx]/1e6:8.3f}     "
            f"{dm_dts[idx]:10.1f}     {altitudes[idx]:7.2f}"
        )

    # -----------------------------
    # Plots for visual inspection
    # -----------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Mach vs time with event markers
    axes[0].plot(times, machs, label="Mach number")
    if booster_fuel_empty_time is not None:
        axes[0].axvline(booster_fuel_empty_time, color="k", linestyle="--", alpha=0.5, label="Booster fuel empty")
    if booster_off_time is not None:
        axes[0].axvline(booster_off_time, color="0.3", linestyle="--", alpha=0.5, label="Booster engine off")
    if separation_time is not None:
        axes[0].axvline(separation_time, color="r", linestyle="--", alpha=0.5, label="Separation")
    if rocket.upper_ignition_start_time is not None:
        ignition_start = rocket.upper_ignition_start_time
        axes[0].axvline(ignition_start, color="g", linestyle="--", alpha=0.5, label="Upper ignition start")
        full_time = ignition_start + rocket.upper_engine_ramp_time
        axes[0].axvline(full_time, color="c", linestyle="--", alpha=0.5, label="Upper full thrust")
    axes[0].set_ylabel("Mach")
    axes[0].set_title("Mach vs time (vertical ascent)")
    axes[0].grid(True)
    axes[0].legend()

    # Thrust vs time and stage index
    axes[1].plot(times, thrusts, label="|Thrust| [N]")
    axes[1].set_ylabel("Thrust [N]")
    axes[1].grid(True)
    ax_stage = axes[1].twinx()
    ax_stage.step(times, stages_idx, where="post", color="tab:orange", alpha=0.7, label="Stage index")
    ax_stage.set_ylabel("Stage index")
    # Build a combined legend
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax_stage.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Altitude vs time
    axes[2].plot(times, altitudes)
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Altitude [km]")
    axes[2].set_title("Altitude vs time")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

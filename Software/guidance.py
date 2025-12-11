from __future__ import annotations
from typing import List, Tuple, Callable, Optional
import numpy as np
from Main.state import State
from .config import SoftwareConfig
from Environment.config import EnvironmentConfig

def create_pitch_program_callable(pitch_points: List[Tuple[float, float]], azimuth_deg: Optional[float] = None) -> Callable[[float, object], np.ndarray]:
    """
    Creates a callable pitch program function for the Guidance class.
    
    This function takes a pitch schedule and an azimuth and returns a callable
    that can be used by the simulation's guidance system.

    Parameters
    ----------
    pitch_points : List[Tuple[float, float]]
        A list of (time_s, pitch_angle_deg) tuples defining the pitch profile.
        Angle is degrees from the local horizontal plane (0 deg is horizontal, 90 deg is vertical).
        Points are assumed to be sorted by time for interpolation.
    azimuth_deg : Optional[float]
        The launch azimuth in degrees from East towards North. Defaults to 0 (due East).

    Returns
    -------
    Callable[[float, object], np.ndarray]
        A function suitable for sim.guidance.pitch_program that returns
        the desired thrust direction vector in ECI frame.
    """
    
    times = np.array([p[0] for p in pitch_points], dtype=float)
    angles_deg = np.array([p[1] for p in pitch_points], dtype=float)

    def pitch_program_function(t: float, state: object) -> np.ndarray:
        """
        Calculates the desired thrust direction based on the parameterized pitch profile.
        Angle is degrees from vertical (90 = straight up, 0 = horizontal).
        """
        desired_pitch_rad = np.radians(np.interp(t, times, angles_deg, left=angles_deg[0], right=angles_deg[-1]))

        r_eci = np.asarray(getattr(state, "r_eci", [0, 0, 1]), dtype=float)
        v_eci = np.asarray(getattr(state, "v_eci", [0, 0, 0]), dtype=float)

        r_norm = np.linalg.norm(r_eci)
        if r_norm < 1e-6:
            return np.array([0.0, 0.0, 1.0])

        vertical_dir = r_eci / r_norm

        v_norm = np.linalg.norm(v_eci)
        if v_norm < 1.0:
            return vertical_dir

        # Tangent direction: azimuth-based if provided, else velocity projection.
        if azimuth_deg is not None:
            east = np.cross(np.array([0.0, 0.0, 1.0]), vertical_dir)
            if np.linalg.norm(east) == 0.0:
                east = np.array([1.0, 0.0, 0.0])
            east = east / np.linalg.norm(east)
            north = np.cross(vertical_dir, east)
            north = north / np.linalg.norm(north)
            az = np.radians(azimuth_deg)
            tangent_dir = np.cos(az) * east + np.sin(az) * north
        else:
            horizontal_dir_raw = v_eci - np.dot(v_eci, vertical_dir) * vertical_dir
            horizontal_norm = np.linalg.norm(horizontal_dir_raw)
            if horizontal_norm < 1e-6:
                if vertical_dir[2] < 0.9:
                    tangent_dir = np.array([0.0, 0.0, 1.0]) - np.dot(np.array([0.0, 0.0, 1.0]), vertical_dir) * vertical_dir
                else:
                    tangent_dir = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), vertical_dir) * vertical_dir
                tangent_dir = tangent_dir / np.linalg.norm(tangent_dir)
            else:
                tangent_dir = horizontal_dir_raw / horizontal_norm

        thrust_dir_eci = np.sin(desired_pitch_rad) * vertical_dir + np.cos(desired_pitch_rad) * tangent_dir
        norm = np.linalg.norm(thrust_dir_eci)
        return thrust_dir_eci / norm if norm > 0 else vertical_dir

    return pitch_program_function

class StageAwarePitchProgram:
    """
    Interpolates pitch angle schedules based on time, separately for booster and upper stage.
    Schedules are lists of [time_s, angle_deg] pairs, where time is measured from the
    start of that stage (liftoff for booster, upper ignition for upper stage).
    Angle is degrees from the local horizontal (0=horizontal, 90=vertical).
    After the last point, transitions to prograde if speed exceeds the threshold, otherwise holds horizontal.
    """

    def __init__(
        self,
        sw_config: SoftwareConfig,
        env_config: EnvironmentConfig
    ):
        self.booster_time_points, self.booster_angles_rad = self._prep_schedule(sw_config.pitch_program)
        self.upper_time_points, self.upper_angles_rad = self._prep_schedule(sw_config.upper_pitch_program)
        self.prograde_threshold = sw_config.pitch_prograde_speed_threshold
        self.earth_radius = env_config.earth_radius_m

    @staticmethod
    def _prep_schedule(schedule: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        if not schedule:
            return np.array([0.0]), np.array([np.pi / 2])
        sorted_sched = sorted(schedule, key=lambda p: p[0])
        times = np.array([p[0] for p in sorted_sched], dtype=float)
        angles = np.deg2rad([p[1] for p in sorted_sched])
        return times, angles

    def __call__(self, t: float, state: State, t_stage: float | None = None, stage_index: int | None = None) -> np.ndarray:
        r = np.asarray(state.r_eci, dtype=float)
        v = np.asarray(state.v_eci, dtype=float)
        r_norm = np.linalg.norm(r)
        if r_norm == 0.0:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        r_hat = r / r_norm

        # Define local orientation vectors (up, east)
        east = np.cross([0.0, 0.0, 1.0], r_hat)
        east_norm = np.linalg.norm(east)
        east = east / east_norm if east_norm > 0.0 else np.array([1.0, 0.0, 0.0], dtype=float)

        idx = 0 if stage_index is None else int(stage_index)
        time_points = self.booster_time_points if idx == 0 else self.upper_time_points
        angle_points = self.booster_angles_rad if idx == 0 else self.upper_angles_rad

        t_rel = t if t_stage is None else float(t_stage)
        final_time = time_points[-1]

        if t_rel > final_time:
            speed = np.linalg.norm(v)
            if speed > self.prograde_threshold:
                direction = v / speed
            else:
                direction = east
        else:
            pitch_rad = np.interp(t_rel, time_points, angle_points)
            direction = np.cos(pitch_rad) * east + np.sin(pitch_rad) * r_hat

        n = np.linalg.norm(direction)
        return direction / n if n > 0.0 else r_hat

class ParameterizedThrottleProgram:
    """
    Interpolates a throttle schedule for the upper stage based on time since
    ignition. The schedule is a list of [time_sec, throttle_level] pairs.
    """

    def __init__(self, schedule: list[list[float]]):
        self.schedule = sorted(schedule, key=lambda p: p[0])
        self.time_points = np.array([p[0] for p in self.schedule])
        self.throttle_points = np.array([p[1] for p in self.schedule])

    def __call__(self, t: float, state: State) -> float:
        ignition_time = getattr(state, "upper_ignition_start_time", None)
        if ignition_time is None:
            return 0.0  # Not yet ignited

        time_since_ignition = t - ignition_time
        if time_since_ignition < 0:
            return 0.0

        # Interpolate throttle level from the schedule
        throttle = np.interp(
            time_since_ignition, self.time_points, self.throttle_points, left=0.0, right=0.0
        )
        return float(throttle)
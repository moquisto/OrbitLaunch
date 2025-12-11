import numpy as np
from typing import List, Tuple, Callable, Optional

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
    azimuth_rad = np.radians(azimuth_deg if azimuth_deg is not None else 0.0)

    def pitch_program_function(t: float, state: object) -> np.ndarray:
        """
        Calculates the desired thrust direction based on the parameterized pitch profile.
        """
        # Interpolate the desired pitch angle at the current time
        pitch_rad = np.radians(np.interp(t, times, angles_deg, left=angles_deg[0], right=angles_deg[-1]))
        
        # Get current position vector and define local coordinate frame
        r_eci = np.asarray(getattr(state, 'r_eci', [0,0,1]), dtype=float)
        r_norm = np.linalg.norm(r_eci)
        
        if r_norm < 1e-6: # Robustness for r_eci=[0,0,0]
            return np.array([0.0, 0.0, 1.0])
        
        # Establish the East-North-Up (ENU) local coordinate frame
        up_dir = r_eci / r_norm
        
        # Define a stable 'East' vector, avoiding gimbal lock at the poles
        z_axis = np.array([0.0, 0.0, 1.0])
        east_dir = np.cross(z_axis, up_dir)
        east_norm = np.linalg.norm(east_dir)
        if east_norm < 1e-9: # True if r_eci is aligned with z_axis (at a pole)
            # If at a pole, 'East' can be arbitrarily chosen in the xy-plane.
            east_dir = np.array([1.0, 0.0, 0.0])
        else:
            east_dir /= east_norm
            
        north_dir = np.cross(up_dir, east_dir)
        
        # Determine the horizontal direction based on azimuth
        # Azimuth is angle from East towards North
        horizontal_dir = np.cos(azimuth_rad) * east_dir + np.sin(azimuth_rad) * north_dir
        
        # The final thrust vector is a combination of the horizontal and vertical directions
        # based on the interpolated pitch angle.
        thrust_dir = np.cos(pitch_rad) * horizontal_dir + np.sin(pitch_rad) * up_dir
        
        # Normalize for safety, though it should be a unit vector by construction
        thrust_norm = np.linalg.norm(thrust_dir)
        if thrust_norm < 1e-9:
            return up_dir # Fallback to vertical thrust
            
        return thrust_dir / thrust_norm

    return pitch_program_function


def simple_pitch_program(t: float, state: object) -> np.ndarray:
    """
    Minimal pitch program used as a safe default for config-based imports.
    Points thrust along the local vertical.
    """
    r_eci = np.asarray(state.r_eci, dtype=float)
    r_norm = np.linalg.norm(r_eci)
    if r_norm == 0.0:
        return np.array([0.0, 0.0, 1.0])
    return r_eci / r_norm

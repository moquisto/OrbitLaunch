import numpy as np
from typing import List, Tuple, Callable, Optional

def create_pitch_program_callable(pitch_points: List[Tuple[float, float]], azimuth_deg: Optional[float] = None) -> Callable[[float, object], np.ndarray]:
    """
    Creates a callable pitch program function for the Guidance class.
    
    Parameters
    ----------
    pitch_points : List[Tuple[float, float]]
        A list of (time_s, pitch_angle_deg) tuples defining the pitch profile.
        pitch_angle_deg is in degrees relative to the vertical (90 deg is straight up, 0 deg is horizontal).
        Points are assumed to be sorted by time.

    Returns
    -------
    Callable[[float, object], np.ndarray]
        A function suitable for sim.guidance.pitch_program that returns
        the desired thrust direction vector in ECI.
    """
    
    times = np.array([p[0] for p in pitch_points], dtype=float)
    angles_deg = np.array([p[1] for p in pitch_points], dtype=float)

    def pitch_program_function(t: float, state: object) -> np.ndarray:
        """
        Calculates the desired thrust direction based on the parameterized pitch profile.
        """
        # Interpolate the desired pitch angle at the current time
        desired_pitch_deg = np.interp(t, times, angles_deg, left=angles_deg[0], right=angles_deg[-1])
        
        # Convert pitch angle to radians
        desired_pitch_rad = np.radians(desired_pitch_deg)
        
        # Get current position vector from state
        r_eci = np.asarray(state.r_eci, dtype=float)
        r_norm = np.linalg.norm(r_eci)
        
        if r_norm == 0.0: # Should not happen in flight, but for robustness
            return np.array([0.0, 0.0, 1.0])
        
        # Get the 'vertical' direction (away from Earth center)
        vertical_dir = r_eci / r_norm
        
        # The desired thrust vector should be in the plane defined by
        # the velocity vector and the vertical vector, and tilted
        # by desired_pitch_rad from vertical.
        # This is a simplification; a full 3D pitch program is more complex.
        # For a 2D ascent, assuming motion primarily in the x-z plane (or y-z plane)
        # we can define a tangent_dir (horizontal) orthogonal to vertical_dir in the plane of motion.
        
        # For simplicity, let's assume the pitch angle is defined relative to the current
        # 'forward' direction, which is often approximated by the velocity vector,
        # or more precisely by the component of velocity tangential to the sphere.
        # However, the prompt specifies "relative to the vertical".
        # Let's derive a simple thrust vector from vertical.
        
        # For now, let's assume the velocity vector defines the plane of motion.
        # A more robust solution would account for yaw.
        
        v_eci = np.asarray(state.v_eci, dtype=float)
        v_norm = np.linalg.norm(v_eci)
        
        if v_norm < 1.0: # If almost stationary, just go vertical
             return vertical_dir
        
        # Tangential direction: use azimuth if provided; otherwise use velocity projection.
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
            # Calculate horizontal direction in the plane defined by r_eci and v_eci
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
            
        # The thrust vector is a linear combination of vertical_dir and tangent_dir
        # desired_pitch_rad is the angle from vertical_dir towards tangent_dir
        
        # For a pitch angle (alpha) from vertical, the vector is:
        # T = cos(alpha) * vertical_dir + sin(alpha) * tangent_dir
        
        thrust_dir_eci = np.sin(desired_pitch_rad) * vertical_dir + np.cos(desired_pitch_rad) * tangent_dir
        
        return thrust_dir_eci

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

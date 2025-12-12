"""
mission_profile.py

Defines the mission objectives and parameters.
This file is intended to hold key objectives for the mission, such as target orbits,
payload requirements, or specific flight constraints.
"""

from dataclasses import dataclass

@dataclass
class MissionProfile:
    # Example: Define a target orbit altitude
    target_orbit_altitude_m: float = 200_000.0  # meters
    
    # Example: Define a target orbital inclination
    target_inclination_deg: float = 28.5  # degrees
    
    # Add other mission-specific parameters here as needed

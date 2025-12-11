import numpy as np
import pytest
from unittest.mock import Mock, patch

from config import Config
from atmosphere import AtmosphereModel # For type hinting and mocking

def test_config_default_values():
    """Test that Config dataclass initializes with expected default values."""
    cfg = Config()

    # Spot-check a few critical default values
    assert cfg.launch_lat_deg == 26.0
    assert cfg.earth_mu == 3.986_004_418e14
    assert cfg.booster_thrust_vac == 7.6e7
    assert cfg.main_dt_s == 0.05
    assert cfg.max_q_limit == 60_000.0
    assert cfg.use_jet_stream_model is True
    assert cfg.orbit_alt_tol == 500.0

    # Check default_factory fields for type and non-emptiness (detailed content can be checked if needed)
    assert isinstance(cfg.earth_omega_vec, tuple)
    assert len(cfg.pitch_program) > 0
    assert len(cfg.upper_pitch_program) > 0
    assert len(cfg.booster_throttle_program) > 0
    assert len(cfg.upper_stage_throttle_program) > 0
    assert len(cfg.wind_alt_points) > 0
    assert len(cfg.mach_cd_map) > 0

def test_config_custom_values():
    """Test that Config dataclass correctly sets custom provided values."""
    custom_cfg = Config(
        launch_lat_deg=30.0,
        earth_mu=1.0,
        booster_thrust_vac=100.0,
        main_dt_s=0.1,
        max_q_limit=50_000.0,
        use_jet_stream_model=False,
        orbit_alt_tol=100.0,
        earth_omega_vec=(0.0, 0.0, 1.0e-5),
        pitch_program=[[0.0, 90.0]],
        atmosphere_switch_alt_m=90000.0
    )

    assert custom_cfg.launch_lat_deg == 30.0
    assert custom_cfg.earth_mu == 1.0
    assert custom_cfg.booster_thrust_vac == 100.0
    assert custom_cfg.main_dt_s == 0.1
    assert custom_cfg.max_q_limit == 50_000.0
    assert custom_cfg.use_jet_stream_model is False
    assert custom_cfg.orbit_alt_tol == 100.0
    assert custom_cfg.earth_omega_vec == (0.0, 0.0, 1.0e-5)
    assert custom_cfg.pitch_program == [[0.0, 90.0]]
    assert custom_cfg.atmosphere_switch_alt_m == 90000.0


@patch('config.AtmosphereModel', autospec=True) # Patch the AtmosphereModel in the config module
def test_config_post_init_atmosphere_model_creation(MockAtmosphereModel):
    """Test that __post_init__ correctly creates an AtmosphereModel instance."""
    
    # Create a Config instance
    cfg = Config(
        atmosphere_switch_alt_m=70000.0,
        launch_lat_deg=10.0,
        launch_lon_deg=-20.0,
        atmosphere_f107=130.0,
        atmosphere_f107a=120.0,
        atmosphere_ap=3.0,
    )

    # Assert that AtmosphereModel was called exactly once
    MockAtmosphereModel.assert_called_once_with(
        h_switch=cfg.atmosphere_switch_alt_m,
        lat_deg=cfg.launch_lat_deg,
        lon_deg=cfg.launch_lon_deg,
        f107=cfg.atmosphere_f107,
        f107a=cfg.atmosphere_f107a,
        ap=cfg.atmosphere_ap,
    )
    # Assert that the _atmosphere_model attribute is set to the mocked instance
    assert cfg._atmosphere_model is MockAtmosphereModel.return_value

@patch('config.AtmosphereModel', autospec=True)
def test_config_get_speed_of_sound_delegation(MockAtmosphereModel):
    """Test that get_speed_of_sound delegates to the internal atmosphere model."""
    cfg = Config()
    
    # Get the mocked instance from the patch
    mock_atm_instance = MockAtmosphereModel.return_value
    
    # Configure the mocked method's return value
    mock_atm_instance.get_speed_of_sound.return_value = 340.29 # Example speed of sound
    
    # Call the method on the Config instance
    altitude = 1000.0
    t = 50.0
    speed_of_sound = cfg.get_speed_of_sound(altitude, t)

    # Assert that the internal atmosphere model's method was called correctly
    mock_atm_instance.get_speed_of_sound.assert_called_once_with(altitude, t)
    
    # Assert that the Config method returned the mocked value
    assert speed_of_sound == 340.29

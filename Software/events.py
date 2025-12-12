# Software/events.py

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from Main.state import State
    from Hardware.rocket import Rocket
    from Software.guidance import GuidanceCommand

class EventManager:
    """
    Manages and executes events triggered during the simulation,
    such as stage separation.
    """
    def __init__(self, rocket: Rocket):
        self.rocket = rocket

    def apply_events(self, state: State, guidance_command: GuidanceCommand) -> State:
        """
        Applies the effects of guidance-commanded events to the simulation state.

        Parameters
        ----------
        state : State
            The current simulation state.
        guidance_command : GuidanceCommand
            The command issued by the guidance system, potentially including
            event triggers like stage separation.

        Returns
        -------
        State
            The updated simulation state after applying event effects.
        """
        if guidance_command.initiate_stage_separation:
            if guidance_command.new_stage_index is None:
                raise ValueError("Guidance commanded separation but new_stage_index is not provided.")
            if guidance_command.dry_mass_to_drop is None:
                raise ValueError("Guidance commanded separation but dry_mass_to_drop is not provided.")

            # Update the state for stage separation
            state.stage_index = guidance_command.new_stage_index
            state.m -= guidance_command.dry_mass_to_drop
            
            # Optionally, reset the rocket's internal state if needed for the new stage
            # (e.g., if any _last_time variables are stage-specific, though currently they are not)
            # self.rocket.reset_stage_state(state.stage_index) # This would require adding such a method to Rocket

            # Log the event for telemetry (future improvement)
            print(f"DEBUG: Stage separation initiated, new stage_index={state.stage_index}, mass dropped={guidance_command.dry_mass_to_drop} kg")

        # Other events can be added here
        # E.g., if guidance_command.deploy_fairings: ...

        return state

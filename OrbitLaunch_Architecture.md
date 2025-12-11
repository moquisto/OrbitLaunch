# 6 folder structure
## Environment

gravity.py
atmosphere.py
aerodynamics.py

These are built once even when running an optimization program that executes multiple simulations
## Hardware
rocket.py
engine.py
stage.py

These will have no executive functions - all software should be in the software folder - no dynamic staging from rocket behavior - only report. For example, if the fuel is low, don't execute staging in the rocket file, instead that will be handled in the guidance file in the software folder. Then there will be functions in the staging.py that executes the staging behavior once it has been determined from the events file.
## Software

guidance.py
mission_profile.py
events.py

Hold all of the guidance logic and specfic actions to take in case of certain events like launch and staging. Eventually perhaps multiple stages can be created, for example to fine-tune the orbit, that is not necessary now though. Mission profile holds the key objectives for the mission things of that sort.
## Main

integrators.py
simulation.py
state.py
telemetry.py

self explanatory, telemetry holds observables like energy and end results - fuel, orbit accuracy and stuff of similar sort. The thigns that we eventuall want to plot. Perhaps it should also have error calculation? Maybe a separate file for error handling.
## Analysis

cost_functions.py
plotting.py
optimization.py (maybe include many different optimization files for different schemes to show convergence of different methods)
generate_logs.py

This is where we run the final simulations to optimize the paths according to whatever goals we have for the mission - should be really basic really like the orbit and the start location. The rest will be optimized by the program. Then we can execute the plotting once these are done. Perhaps another thing to do is logging. That should be a separate folder though - but good since that allows us to keep record of old results and the process.

## Logging

Should hold logs for the optimization results and telemetry data for the specific run we did (cost_function, start location, end orbit)
Each run of the optimization script should at the end execute the generate_logs file which creates a new file, likely csv or something with a name based on numbering maybe. 

# CARLA Unified Dashboard

This repository contains a collection of CARLA client tools and a unified dashboard application to run and inspect CARLA simulation sessions.

Primary app file (runs the dashboard and drives the system):

C:\WindowsNoEditor\PythonAPI\examples\carla_unified_dashboard (1).py

(There is also a cleaned copy: C:\WindowsNoEditor\PythonAPI\examples\carla_unified_dashboard.py)

## What this repo contains
- Example clients and tools for CARLA under `examples/`.
- CARLA Python API helpers under `carla/agents/` and related modules.
- Utility scripts under `util/`.

## Overview of the main dashboard script
The file `examples/carla_unified_dashboard (1).py` is the main interactive CARLA dashboard for this repo. It is a single-file application that combines a live Pygame UI, CARLA driving controls, CSV data logging, and an automatic replay/AI comparison workflow.

In practical terms, the script does the following:
- Connects to a running CARLA server on localhost:2000.
- Spawns a vehicle and attaches a camera, collision sensor, and lane-invasion sensor.
- Lets you drive manually with keyboard controls while logging speed, steering, throttle, brake, GPS/position, weather, NPC counts, and collision/lane events.
- Saves the recorded driving session to `driving_data.csv` for later replay or training.
- Trains a small scikit-learn Random Forest model from the saved CSV data to estimate risk and safe control behavior.
- Replays a recorded route in Auto mode using waypoint-following logic and smooth control correction so the AI can follow the human path without teleporting the vehicle.
- Compares the original manual driving data to AI-corrected motion in a side-by-side dashboard panel and highlights risk frames (collision, lane drift, overspeeding, sharp steering, etc.).

### Manual mode
The manual workflow is meant for creating a clean training dataset. Once the car is connected and spawned, you can drive using WASD or arrow keys, switch camera views, change weather, and log each frame of data. Every 10 frames, the script writes a row to the CSV with timestamped telemetry and vehicle pose. This dataset becomes the basis for auto-replay and ML training.

### Auto mode
The auto workflow reads the CSV back in, interpolates the recorded GPS waypoints into a dense route, and attempts to follow them with a PID-style waypoint follower. It smooths the steering, enforces speed caps, anticipates collisions and sharp turns, and reacts to nearby traffic. This lets the system replay a prior human drive with AI assistance and compare it against the original performance.

### Model and analysis pieces
The script is structured around a few key components:
- `CarlaManager`: connection, vehicle spawning, camera/sensor setup, weather changes, traffic management.
- `DataLogger`: writes CSV rows with the recorded simulation state.
- `DrivingModel`: trains and loads a Random Forest model for safety prediction.
- `WaypointFollower`: follows GPS waypoints in auto mode with smoothing, PID steering, speed cap logic, and stuck-state recovery.
- `ComparisonAnalyser`: estimates risk events and summarizes human-vs-AI driving behavior.
- `UnifiedDashboard`: the Pygame interface that orchestrates menu, manual, and auto modes.

### Typical usage
1. Start the CARLA simulator server.
2. Install the dependencies listed in the project requirements.
3. Run the script from the examples folder.
4. Press CONNECT and SPAWN CAR.
5. Drive manually in MANUAL mode to record a route.
6. Switch to AUTO mode to replay the stored path and compare the AI-corrected version to the original data.

There are two variants of this app in the repo: `carla_unified_dashboard (1).py` and the cleaned copy `carla_unified_dashboard.py`. The numbered version is the fuller, feature-rich dashboard script described above.

## Quickstart (recommended)
1. Install Python 3.7 (this project was developed and tested with Python 3.7).
2. Create and activate a virtual environment:

   python -m venv .venv
   .venv\Scripts\activate    (Windows)

3. Install requirements. There are requirements files in several folders; example:

   pip install -r examples/requirements.txt
   pip install -r util/requirements.txt

4. Configure environment variables:

   copy .env.example .env
   (edit .env to match your CARLA host/port and display settings)

5. Start the CARLA server (Unreal + CARLA) on the same machine or on the host specified in .env.
6. Run the dashboard app (from the examples folder):

   py -3.7 "examples\carla_unified_dashboard (1).py"

   or

   py -3.7 examples\carla_unified_dashboard.py

## Notes
- Large artifacts, Python wheels, build/distribution folders, and model files are intentionally excluded from the repository. Keep model binaries, CARLA builds, and Unreal assets outside the git repo and add references or download instructions instead.
- If you need to include a trained model, add a small sample or provide a download link and keep the real binary out of git.

## Contributing
If you'd like assistance pushing the full source tree (excluding large binaries) into the repository as a single commit, grant permission and the upload will proceed. Alternatively, follow the steps above locally and push using your normal git workflow.

## License
Add license text or a LICENSE file if needed.

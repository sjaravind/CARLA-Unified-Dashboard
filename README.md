# CARLA Unified Dashboard

This repository contains a collection of CARLA client tools and a unified dashboard application to run and inspect CARLA simulation sessions.

Primary app file (runs the dashboard and drives the system):

C:\WindowsNoEditor\PythonAPI\examples\carla_unified_dashboard (1).py

(There is also a cleaned copy: C:\WindowsNoEditor\PythonAPI\examples\carla_unified_dashboard.py)

## What this repo contains
- Example clients and tools for CARLA under `examples/`.
- CARLA Python API helpers under `carla/agents/` and related modules.
- Utility scripts under `util/`.

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

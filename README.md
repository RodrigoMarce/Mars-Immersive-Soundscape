# Mars Immersive Soundscape

This repository contains code for a real-time pose tracking and energy
analysis system built around YOLOv8 and MediaPipe. The project captures
people via a webcam, tracks each person, estimates an "energy" level
based on limb movement, and sends results over OSC for use in an
immersive soundscape or other interactive installation.

## Features

- **Person detection** using YOLOv8 (`yolov8n.pt`).
- **Pose estimation** via MediaPipe's lightweight Pose Landmarker model
  (`pose_landmarker_lite.task`).
- **Tracking** of individuals across frames with simple IOU-based
  association.
- **Energy computation** derived from wrist, ankle and hip movement;
  mapped to discrete levels (`idle`, `light`, `moderate`, `high`).
- **OSC output** for people count, group statistics and per-person
  energy using `python-osc`.

## Repository Layout

```text
Mars-Immersive-Soundscape/
├── helpers.py           # Utility functions (IOU, energy math, etc.)
├── osc_client.py        # OSC wrapper
├── pose_actions_multipeople.py  # Main demo script
├── tracker.py           # Simple tracker implementation
├── requirements.txt     # Python dependencies
├── yolov8n.pt           # YOLO model weights (git-lfs or download separately)
└── pose_landmarker_lite.task  # MediaPipe pose model
```

## Cloning the Repository

If you don't already have the code locally, clone the GitHub project:

```bash
git clone https://github.com/RodrigoMarce/Mars-Immersive-Soundscape.git

cd Mars-Immersive-Soundscape
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure you have a webcam connected and working.

## Usage

Run the main script:

```bash
python pose_actions_multipeople.py
```

- Press `q` in the display window to quit.
- OSC messages are emitted to `127.0.0.1:8000` by default; adjust the
  constants in `pose_actions_multipeople.py` or pass parameters to
  `OSCClient` if needed.

> **Note:** This Python script is designed to work in tandem with a
> Max/MSP patch that receives the OSC data and drives the immersive
> soundscape. Be sure the Max patch is running and listening on the same
> IP/port before starting the Python program; otherwise, you won't hear
> the audio output and the system won't function as a complete
> installation.

## Extending the Project

You can modify the energy computation thresholds, add new OSC
endpoints, or integrate additional models. The `helpers.py` file
contains most of the math utilities used by other modules.

## License

This code is provided as-is for educational and demonstration purposes.
Feel free to adapt it for your own projects.

# Niryo Ned2 Vision-Guided Pick-and-Place

A computer-vision-guided robotic pick-and-place system built using a Niryo Ned2 and an M5Stack UnitV camera.

The UnitV detects a red cylinder and sends its pixel coordinates to a Python program through serial communication. A calibrated quadratic mapping converts the camera coordinates into robot coordinates, allowing the Ned2 to locate, pick up, lift and return the cylinder automatically.

## Features

- Real-time red-object detection
- Bottom-centre object tracking
- USB serial communication
- Nine-point camera-to-robot calibration
- Quadratic coordinate mapping
- Automatic pick, lift, return and release
- Camera-stability and workspace checks
- Confused-search and trophy-celebration motions

## Hardware

- Niryo Ned2
- Large Gripper
- M5Stack UnitV camera
- Windows laptop
- Red foam cylinder

## Software

- Python
- PyNiryo
- NumPy
- PySerial
- MaixPy IDE
- NiryoStudio

## Files

- `unitv_red_tracker.py`: Detects the red cylinder and outputs its bottom-centre coordinates.
- `test_hover.py`: Tests camera-to-robot positioning at a safe height.
- `test_descent.py`: Tests the staged vertical approach.
- `auto_pick_and_return.py`: Picks up and returns the cylinder automatically.
- `auto_pick_showoff.py`: Adds confused-search and trophy-celebration motions.
- `manual_calibration_points.csv`: Camera and robot calibration measurements.

## Installation

```bash
py -m pip install -r requirements.txt
Usage

Run the UnitV tracker and disconnect MaixPy IDE so the camera serial port is available.

Then run:

py auto_pick_showoff.py


Important

The calibration values are specific to the camera and robot placement used during development. Moving the camera requires recalibration.

Always operate the robot in a clear workspace with the emergency stop accessible.

Author - Layan Barakat

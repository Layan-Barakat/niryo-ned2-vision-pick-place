"""Guarded camera-guided pickup for a Niryo Ned2.

The initial version accepts U,V manually from the MC4 terminal. Direct serial
reading can be added after the camera firmware/program continuously publishes a
simple line such as "UV:196.0,155.5".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from pyniryo import NiryoRobot, ToolID


CALIBRATION_FILE = Path(__file__).with_name("calibration.json")
TRANSIT_SPEED_PERCENT = 50
APPROACH_SPEED_PERCENT = 20
HOVER_CLEARANCE_M = 0.10
PREGRASP_CLEARANCE_M = 0.03
WORKSPACE_MARGIN_MM = 15.0


def parse_uv(text: str) -> tuple[float, float]:
    cleaned = text.replace("(", "").replace(")", "").replace(";", ",")
    pieces = [piece.strip() for piece in cleaned.split(",") if piece.strip()]
    if len(pieces) != 2:
        pieces = cleaned.split()
    if len(pieces) != 2:
        raise ValueError("Enter two numbers, for example: 196, 155.5")
    return float(pieces[0]), float(pieces[1])


def transform_uv(matrix: np.ndarray, u: float, v: float) -> tuple[float, float]:
    mapped = matrix @ np.asarray([u, v, 1.0])
    if abs(mapped[2]) < 1e-9:
        raise RuntimeError("Invalid transformation denominator; do not move the robot")
    x_mm = float(mapped[0] / mapped[2])
    y_mm = float(mapped[1] / mapped[2])
    if not (math.isfinite(x_mm) and math.isfinite(y_mm)):
        raise RuntimeError("Transformation returned a non-finite coordinate; do not move the robot")
    return x_mm, y_mm


def confirmed(prompt: str) -> bool:
    return input(f"{prompt} Type YES to continue: ").strip() == "YES"


def main() -> None:
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError("Run calibrate.py first; calibration.json does not exist")

    calibration = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    matrix = np.asarray(calibration["matrix_uv_to_xy_mm"], dtype=float)
    bounds = calibration["workspace_bounds_mm"]
    pick_pose = calibration["suggested_pick_pose"]

    u, v = parse_uv(input("Fresh camera U,V from centre(): ").strip())
    x_mm, y_mm = transform_uv(matrix, u, v)
    print(f"Predicted robot target: X={x_mm:.2f} mm, Y={y_mm:.2f} mm")

    inside_x = bounds["x_min"] - WORKSPACE_MARGIN_MM <= x_mm <= bounds["x_max"] + WORKSPACE_MARGIN_MM
    inside_y = bounds["y_min"] - WORKSPACE_MARGIN_MM <= y_mm <= bounds["y_max"] + WORKSPACE_MARGIN_MM
    if not (inside_x and inside_y):
        raise RuntimeError("Predicted target is outside the calibrated work area; movement blocked")

    pick_z = float(pick_pose["z_m"])
    roll = float(pick_pose["roll_rad"])
    pitch = float(pick_pose["pitch_rad"])
    yaw = float(pick_pose["yaw_rad"])
    hover_z = pick_z + HOVER_CLEARANCE_M
    pregrasp_z = pick_z + PREGRASP_CLEARANCE_M
    x_m, y_m = x_mm / 1000.0, y_mm / 1000.0

    print(f"Pickup Z from calibration: {pick_z * 1000:.1f} mm")
    print(f"First destination is the safe hover: Z={hover_z * 1000:.1f} mm")
    if not confirmed("Clear the workspace and keep the emergency stop accessible."):
        print("Cancelled before connecting to the robot.")
        return

    robot = NiryoRobot(calibration.get("robot_ip", "169.254.200.200"))
    try:
        tool_id = robot.get_current_tool_id()
        print(f"Active tool: {tool_id}")
        if tool_id == ToolID.NONE:
            raise RuntimeError(
                "No active gripper. Connect NiryoStudio, select Large Gripper, "
                "and do not call update_tool()."
            )

        robot.set_arm_max_velocity(TRANSIT_SPEED_PERCENT)
        robot.open_gripper()
        robot.move_pose(x_m, y_m, hover_z, roll, pitch, yaw)
        print("Hover reached. Visually check alignment from two directions.")

        if not confirmed("Is the gripper centred above the cylinder?"):
            print("Stopped safely at hover; no descent performed.")
            return

        robot.set_arm_max_velocity(APPROACH_SPEED_PERCENT)
        robot.move_pose(x_m, y_m, pregrasp_z, roll, pitch, yaw)
        print("Pre-grasp height reached.")
        if not confirmed("Is the final vertical descent clear and correctly aligned?"):
            print("Stopped safely at pre-grasp height.")
            return

        robot.move_pose(x_m, y_m, pick_z, roll, pitch, yaw)
        robot.close_gripper()
        print("Gripper closed. Beginning vertical lift.")
        robot.set_arm_max_velocity(TRANSIT_SPEED_PERCENT)
        robot.move_pose(x_m, y_m, hover_z, roll, pitch, yaw)
        print("Pickup sequence complete. The robot remains at hover.")
    finally:
        robot.close_connection()


if __name__ == "__main__":
    main()

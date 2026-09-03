"""Interactive camera-to-Ned2 planar calibration helper.

For each calibration point:
1. Put the cylinder at a different location in the marked work area.
2. Read its camera centre (U, V) from the MC4 terminal.
3. Jog the open gripper so its centre is aligned with the cylinder.
4. Enter U,V here; this script reads the robot pose automatically.

Only X and Y are fitted. Z and orientation are stored as suggested pickup
defaults, but should be checked physically before the first real pickup.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pyniryo import NiryoRobot


ROBOT_IP = "169.254.200.200"
MIN_POINTS = 6
DEFAULT_POINTS = 8
OUTPUT_JSON = Path(__file__).with_name("calibration.json")
OUTPUT_CSV = Path(__file__).with_name("calibration_points.csv")


def pose_values(pose) -> list[float]:
    """Convert the PyNiryo pose object into [x, y, z, roll, pitch, yaw]."""
    if hasattr(pose, "to_list"):
        values = pose.to_list()
    elif all(hasattr(pose, name) for name in ("x", "y", "z", "roll", "pitch", "yaw")):
        values = [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]
    else:
        values = list(pose)
    return [float(value) for value in values[:6]]


def parse_uv(text: str) -> tuple[float, float]:
    cleaned = text.replace("(", "").replace(")", "").replace(";", ",")
    pieces = [piece.strip() for piece in cleaned.split(",") if piece.strip()]
    if len(pieces) != 2:
        pieces = cleaned.split()
    if len(pieces) != 2:
        raise ValueError("Enter two numbers, for example: 196, 155.5")
    return float(pieces[0]), float(pieces[1])


def fit_homography(points: list[dict]) -> np.ndarray:
    """Least-squares homography mapping camera pixels (u,v) to robot mm (x,y)."""
    rows: list[list[float]] = []
    targets: list[float] = []
    for point in points:
        u, v = point["u"], point["v"]
        x, y = point["robot_x_mm"], point["robot_y_mm"]
        rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x])
        targets.append(x)
        rows.append([0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y])
        targets.append(y)
    solution, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(targets), rcond=None)
    return np.append(solution, 1.0).reshape(3, 3)


def transform_uv(matrix: np.ndarray, u: float, v: float) -> tuple[float, float]:
    mapped = matrix @ np.asarray([u, v, 1.0])
    if abs(mapped[2]) < 1e-9:
        raise ZeroDivisionError("Calibration produced an invalid projective denominator")
    return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])


def calibration_errors(matrix: np.ndarray, points: list[dict]) -> list[float]:
    errors = []
    for point in points:
        predicted_x, predicted_y = transform_uv(matrix, point["u"], point["v"])
        errors.append(
            float(
                np.hypot(
                    predicted_x - point["robot_x_mm"],
                    predicted_y - point["robot_y_mm"],
                )
            )
        )
    return errors


def main() -> None:
    print("\nNed2 camera calibration helper")
    print("This script reads robot poses only; it does not command arm movement.")
    print("Keep NiryoStudio connected so the Large Gripper remains selected.\n")

    requested = input(f"Number of calibration points [{DEFAULT_POINTS}]: ").strip()
    count = int(requested) if requested else DEFAULT_POINTS
    if count < MIN_POINTS:
        raise ValueError(f"Use at least {MIN_POINTS} points for a useful calibration")

    robot = NiryoRobot(ROBOT_IP)
    points: list[dict] = []
    try:
        while len(points) < count:
            index = len(points) + 1
            print(f"\n--- Point {index}/{count} ---")
            print("Place the cylinder, run centre() on the MC4, then centre the gripper over it.")
            while True:
                try:
                    u, v = parse_uv(input("Camera U,V: ").strip())
                    break
                except ValueError as error:
                    print(error)

            pose = pose_values(robot.get_pose())
            x_m, y_m, z_m, roll, pitch, yaw = pose
            print(
                "Robot pose read: "
                f"X={x_m * 1000:.3f} mm, Y={y_m * 1000:.3f} mm, "
                f"Z={z_m * 1000:.3f} mm"
            )
            keep = input("Record this pair? [Y/n]: ").strip().lower()
            if keep in {"n", "no"}:
                print("Point discarded; reposition and enter it again.")
                continue

            points.append(
                {
                    "u": u,
                    "v": v,
                    "robot_x_mm": x_m * 1000.0,
                    "robot_y_mm": y_m * 1000.0,
                    "robot_z_m": z_m,
                    "roll_rad": roll,
                    "pitch_rad": pitch,
                    "yaw_rad": yaw,
                }
            )
    finally:
        robot.close_connection()

    if len(points) < MIN_POINTS:
        raise RuntimeError(f"Only {len(points)} points were recorded; at least {MIN_POINTS} are required")

    matrix = fit_homography(points)
    errors = calibration_errors(matrix, points)
    rms_error = float(np.sqrt(np.mean(np.square(errors))))
    max_error = float(max(errors))

    z_values = [point["robot_z_m"] for point in points]
    roll_values = [point["roll_rad"] for point in points]
    pitch_values = [point["pitch_rad"] for point in points]
    yaw_values = [point["yaw_rad"] for point in points]
    x_values = [point["robot_x_mm"] for point in points]
    y_values = [point["robot_y_mm"] for point in points]

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "robot_ip": ROBOT_IP,
        "matrix_uv_to_xy_mm": matrix.tolist(),
        "rms_training_error_mm": rms_error,
        "max_training_error_mm": max_error,
        "point_count": len(points),
        "workspace_bounds_mm": {
            "x_min": min(x_values),
            "x_max": max(x_values),
            "y_min": min(y_values),
            "y_max": max(y_values),
        },
        "suggested_pick_pose": {
            "z_m": float(np.median(z_values)),
            "roll_rad": float(np.median(roll_values)),
            "pitch_rad": float(np.median(pitch_values)),
            "yaw_rad": float(np.median(yaw_values)),
        },
        "points": points,
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    print("\nCalibration complete")
    print(f"RMS training error: {rms_error:.2f} mm")
    print(f"Worst training error: {max_error:.2f} mm")
    for index, error in enumerate(errors, start=1):
        print(f"  Point {index}: {error:.2f} mm")
    print(f"Saved: {OUTPUT_JSON.name}")
    print(f"Saved: {OUTPUT_CSV.name}")
    if rms_error > 5.0 or max_error > 10.0:
        print("WARNING: Error is high. Redo inaccurate points before running a pickup.")


if __name__ == "__main__":
    main()

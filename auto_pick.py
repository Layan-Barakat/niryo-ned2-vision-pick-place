import math
import re
import time

import numpy as np
import serial
from pyniryo import NiryoRobot


ROBOT_IP = "169.254.200.200"
CAMERA_PORT = "COM9"
CAMERA_BAUD = 115200

HOVER_Z = 0.240
PREGRASP_Z = 0.120
PICK_Z = 0.075

TRAVEL_SPEED = 50
PICK_SPEED = 20

# U, V, robot X (mm), robot Y (mm)
POINTS = np.array([
    [217, 182,  55.886, -396.978],
    [225, 193,   5.496, -399.614],
    [238, 211, -72.833, -393.933],
    [176, 213, -68.441, -345.666],
    [ 74, 213, -96.464, -249.617],
    [ 90, 193, -32.247, -246.169],
    [111, 188,  28.088, -264.696],
    [158, 186,  33.392, -310.800],
    [164, 197, -24.410, -325.094],
], dtype=float)


def fit_quadratic():
    u = POINTS[:, 0]
    v = POINTS[:, 1]

    camera_features = np.column_stack([
        u,
        v,
        u ** 2,
        u * v,
        v ** 2,
        np.ones(len(u)),
    ])

    robot_positions = POINTS[:, 2:4]

    coefficients, *_ = np.linalg.lstsq(
        camera_features,
        robot_positions,
        rcond=None,
    )

    return coefficients


def read_camera_position(sample_count=10):
    print(f"Connecting to camera on {CAMERA_PORT}...")

    try:
        camera = serial.Serial(
            CAMERA_PORT,
            CAMERA_BAUD,
            timeout=1,
        )
    except serial.SerialException as error:
        raise RuntimeError(
            "Could not open COM9. Disconnect MaixPy IDE from the camera first."
        ) from error

    samples = []
    deadline = time.time() + 15

    try:
        camera.reset_input_buffer()

        while len(samples) < sample_count and time.time() < deadline:
            line = camera.readline().decode(
                "utf-8",
                errors="ignore",
            ).strip()

            match = re.search(
                r"Cylinder U,V:\s*(\d+)\s+(\d+)",
                line,
            )

            if match:
                u = int(match.group(1))
                v = int(match.group(2))

                samples.append((u, v))
                print(f"Camera reading: U={u}, V={v}")

    finally:
        camera.close()

    if len(samples) < sample_count:
        raise RuntimeError(
            "The camera did not provide enough readings."
        )

    u_values = [sample[0] for sample in samples]
    v_values = [sample[1] for sample in samples]

    if max(u_values) - min(u_values) > 5:
        raise RuntimeError(
            "Camera U readings are unstable. Keep the cylinder still."
        )

    if max(v_values) - min(v_values) > 5:
        raise RuntimeError(
            "Camera V readings are unstable. Keep the cylinder still."
        )

    u = float(np.median(u_values))
    v = float(np.median(v_values))

    return u, v


def camera_to_robot(u, v, coefficients):
    camera_point = np.array([
        u,
        v,
        u ** 2,
        u * v,
        v ** 2,
        1,
    ])

    return camera_point @ coefficients


def move_robot():
    coefficients = fit_quadratic()

    u, v = read_camera_position()

    print()
    print(f"Stable camera position: U={u:.0f}, V={v:.0f}")

    if not (74 <= u <= 238 and 182 <= v <= 213):
        raise RuntimeError(
            "The cylinder is outside the calibrated camera area."
        )

    x_mm, y_mm = camera_to_robot(u, v, coefficients)

    print(
        f"Predicted robot position: "
        f"X={x_mm:.3f} mm, Y={y_mm:.3f} mm"
    )

    confirmation = input(
        "Clear the robot's path. Type YES to begin: "
    )

    if confirmation.strip() != "YES":
        print("Cancelled.")
        return

    robot = NiryoRobot(ROBOT_IP)

    try:
        robot.open_gripper()
        robot.set_arm_max_velocity(TRAVEL_SPEED)

        # Move above the cylinder.
        robot.move_pose(
            x_mm / 1000,
            y_mm / 1000,
            HOVER_Z,
            math.radians(-5.121),
            math.radians(1.667),
            math.radians(-90.208),
        )

        print("Hover position reached.")

        confirmation = input(
            "Is the open gripper centred above the cylinder? "
            "Type YES to perform the pickup: "
        )

        if confirmation.strip() != "YES":
            print("Stopped safely at hover.")
            return

        robot.set_arm_max_velocity(PICK_SPEED)

        # Approach the cylinder.
        robot.move_pose(
            x_mm / 1000,
            y_mm / 1000,
            PREGRASP_Z,
            math.radians(-5.121),
            math.radians(1.667),
            math.radians(-90.208),
        )

        # Move into the tested pickup position.
        robot.move_pose(
            x_mm / 1000,
            y_mm / 1000,
            PICK_Z,
            math.radians(-5.121),
            math.radians(1.667),
            math.radians(-90.208),
        )

        # Grab the cylinder.
        robot.close_gripper()
        print("Gripper closed.")

        # Lift vertically.
        robot.set_arm_max_velocity(TRAVEL_SPEED)

        robot.move_pose(
            x_mm / 1000,
            y_mm / 1000,
            HOVER_Z,
            math.radians(-5.121),
            math.radians(1.667),
            math.radians(-90.208),
        )

        print("PICKUP COMPLETE! Cylinder lifted.")

    finally:
        robot.close_connection()


if __name__ == "__main__":
    move_robot()
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
TROPHY_Z = 0.260

TRAVEL_SPEED = 60
PICK_SPEED = 40

ROLL_DEG = -5.121
PITCH_DEG = 1.667
YAW_DEG = -90.208

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

    features = np.column_stack([
        u,
        v,
        u ** 2,
        u * v,
        v ** 2,
        np.ones(len(u)),
    ])

    robot_xy = POINTS[:, 2:4]

    coefficients, *_ = np.linalg.lstsq(
        features,
        robot_xy,
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
            "Could not open COM9. Disconnect MaixPy IDE first."
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
            "Camera U readings are unstable. Movement blocked."
        )

    if max(v_values) - min(v_values) > 5:
        raise RuntimeError(
            "Camera V readings are unstable. Movement blocked."
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


def move_to(robot, x_mm, y_mm, z_m, yaw_deg=YAW_DEG):
    robot.move_pose(
        x_mm / 1000,
        y_mm / 1000,
        z_m,
        math.radians(ROLL_DEG),
        math.radians(PITCH_DEG),
        math.radians(yaw_deg),
    )


def confused_search(robot):
    print("Robot: Where is the cylinder? 🤔")

    # Move safely left and right as though searching.
    move_to(robot, -20, -325, 0.260)
    move_to(robot, -20, -275, 0.260)
    move_to(robot, -20, -300, 0.260)

    print("Robot: Target found!")


def trophy_celebration(robot, x_mm, y_mm):
    print("Robot: Look what I found! 🏆")

    # Raise the cylinder.
    move_to(robot, x_mm, y_mm, TROPHY_Z)

    # Small wrist celebration.
    move_to(robot, x_mm, y_mm, TROPHY_Z, yaw_deg=-80)
    move_to(robot, x_mm, y_mm, TROPHY_Z, yaw_deg=-100)
    move_to(robot, x_mm, y_mm, TROPHY_Z, yaw_deg=YAW_DEG)

    time.sleep(1)


def main():
    coefficients = fit_quadratic()
    u, v = read_camera_position()

    print()
    print(f"Stable camera position: U={u:.0f}, V={v:.0f}")

    if not (74 <= u <= 238 and 182 <= v <= 213):
        raise RuntimeError(
            "The cylinder is outside the calibrated area. Movement blocked."
        )

    x_mm, y_mm = camera_to_robot(u, v, coefficients)

    print(
        f"Predicted position: "
        f"X={x_mm:.3f} mm, Y={y_mm:.3f} mm"
    )

    print()
    print("Automatic show begins in 3 seconds.")
    print("Press Ctrl+C now to cancel.")

    for seconds in range(3, 0, -1):
        print(seconds)
        time.sleep(1)

    robot = NiryoRobot(ROBOT_IP)

    try:
        robot.open_gripper()
        robot.set_arm_max_velocity(TRAVEL_SPEED)

        # Confused motion happens first.
        confused_search(robot)

        # Move above the detected cylinder.
        move_to(robot, x_mm, y_mm, HOVER_Z)
        print("Hover reached.")

        # Descend and grab it.
        robot.set_arm_max_velocity(PICK_SPEED)

        move_to(robot, x_mm, y_mm, PREGRASP_Z)
        move_to(robot, x_mm, y_mm, PICK_Z)

        robot.close_gripper()
        print("Cylinder grabbed!")

        # Lift it.
        robot.set_arm_max_velocity(TRAVEL_SPEED)
        move_to(robot, x_mm, y_mm, HOVER_Z)

        # Trophy celebration happens second.
        trophy_celebration(robot, x_mm, y_mm)

        # Return the cylinder to its original position.
        robot.set_arm_max_velocity(PICK_SPEED)

        move_to(robot, x_mm, y_mm, PREGRASP_Z)
        move_to(robot, x_mm, y_mm, PICK_Z)

        robot.open_gripper()
        print("Cylinder returned.")

        time.sleep(1)

        # Lift away from the cylinder.
        robot.set_arm_max_velocity(TRAVEL_SPEED)
        move_to(robot, x_mm, y_mm, HOVER_Z)

        print("Show complete!")

    finally:
        robot.close_connection()


if __name__ == "__main__":
    main()
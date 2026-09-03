import math
import numpy as np
from pyniryo import NiryoRobot


ROBOT_IP = "169.254.200.200"

HOVER_Z = 0.240
PREGRASP_Z = 0.075

HOVER_SPEED = 50
DESCENT_SPEED = 20

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


# Ask for the latest bottom-centre camera coordinates.
text = input("Enter the cylinder bottom-centre U,V: ")
u, v = map(float, text.replace(",", " ").split())


# Block coordinates outside the calibrated work area.
if not (74 <= u <= 238 and 182 <= v <= 213):
    raise RuntimeError(
        "Camera coordinates are outside the calibrated area. Movement blocked."
    )


# Calculate the robot X and Y position.
coefficients = fit_quadratic()

camera_point = np.array([
    u,
    v,
    u ** 2,
    u * v,
    v ** 2,
    1,
])

x_mm, y_mm = camera_point @ coefficients

print()
print(f"Predicted position: X={x_mm:.3f} mm, Y={y_mm:.3f} mm")
print(f"First height: {HOVER_Z * 1000:.0f} mm")
print(f"Second height: {PREGRASP_Z * 1000:.0f} mm")
print("The gripper will remain open.")


# First safety confirmation.
confirmation = input(
    "Clear the robot's path and keep the emergency stop accessible. "
    "Type YES to begin: "
)

if confirmation.strip() != "YES":
    print("Cancelled.")
    raise SystemExit


robot = NiryoRobot(ROBOT_IP)

try:
    # Keep the gripper open.
    robot.open_gripper()

    # Move to the safe hover position.
    robot.set_arm_max_velocity(HOVER_SPEED)

    robot.move_pose(
        x_mm / 1000,
        y_mm / 1000,
        HOVER_Z,
        math.radians(-5.121),
        math.radians(1.667),
        math.radians(-90.208),
    )

    print()
    print("Safe hover reached.")

    # Second safety confirmation before descending.
    confirmation = input(
        "Is the gripper centred above the cylinder? "
        "Type YES to descend slowly: "
    )

    if confirmation.strip() != "YES":
        print("Stopped safely at the hover position.")
        raise SystemExit

    # Descend slowly to the pre-grasp height.
    robot.set_arm_max_velocity(DESCENT_SPEED)

    robot.move_pose(
        x_mm / 1000,
        y_mm / 1000,
        PREGRASP_Z,
        math.radians(-5.121),
        math.radians(1.667),
        math.radians(-90.208),
    )

    print()
    print("Pre-grasp test complete.")
    print("The gripper is still open and the robot will not descend further.")

finally:
    robot.close_connection()
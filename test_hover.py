import math
import numpy as np
from pyniryo import NiryoRobot


ROBOT_IP = "169.254.200.200"
SPEED = 50
HOVER_Z = 0.240  # 240 mm, safely above the cylinder

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


text = input("Enter the new bottom-centre U,V: ")
u, v = map(float, text.replace(",", " ").split())

if not (74 <= u <= 238 and 182 <= v <= 213):
    raise RuntimeError("Camera coordinates are outside the calibrated area.")

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

print(f"Predicted position: X={x_mm:.3f} mm, Y={y_mm:.3f} mm")
print("The arm will only hover above this position.")

if input("Type YES to move: ").strip() != "YES":
    print("Cancelled.")
    raise SystemExit

robot = NiryoRobot(ROBOT_IP)

try:
    robot.set_arm_max_velocity(SPEED)
    robot.open_gripper()

    robot.move_pose(
        x_mm / 1000,
        y_mm / 1000,
        HOVER_Z,
        math.radians(-5.121),
        math.radians(1.667),
        math.radians(-90.208),
    )

    print("Hover test finished. No descent or grabbing was performed.")

finally:
    robot.close_connection()
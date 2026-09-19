import os
import sys
import time

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pybullet as p
import pybullet_data

from controllers.pd_controller import PDController


STANDING_POSE = np.array(
    [0.0, 0.9, -1.8] * 4,
    dtype=np.float32
)

TORQUE_LIMIT = 33.5


def main():

    client = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    p.loadURDF("plane.urdf")

    robot = p.loadURDF(
        os.path.join(
            pybullet_data.getDataPath(),
            "a1",
            "a1.urdf",
        ),
        [0, 0, 0.35],
        useFixedBase=False,
    )

    revolute_joints = []

    for joint_id in range(p.getNumJoints(robot)):

        info = p.getJointInfo(robot, joint_id)

        if info[2] == p.JOINT_REVOLUTE:
            revolute_joints.append(joint_id)

    print("Revolute joints:", len(revolute_joints))

    # Disable PyBullet's default joint motors.
    for joint_id in revolute_joints:
        p.setJointMotorControl2(
            robot,
            joint_id,
            p.VELOCITY_CONTROL,
            force=0,
        )

    # Initialize exact nominal pose.
    for i, joint_id in enumerate(revolute_joints):

        p.resetJointState(
            robot,
            joint_id,
            float(STANDING_POSE[i]),
            targetVelocity=0,
        )

    controller = PDController(
        kp=40.0,
        kd=1.0,
        torque_limit=TORQUE_LIMIT,
    )

    print()
    print("=" * 70)
    print("PD STANDING EQUILIBRIUM SEARCH")
    print("=" * 70)
    print()

    for step in range(240 * 8):

        joint_states = [
            p.getJointState(robot, joint_id)
            for joint_id in revolute_joints
        ]

        q = np.array(
            [s[0] for s in joint_states],
            dtype=np.float32,
        )

        qd = np.array(
            [s[1] for s in joint_states],
            dtype=np.float32,
        )

        torque = controller.compute_torque(
            STANDING_POSE,
            q,
            qd,
        )

        for i, joint_id in enumerate(revolute_joints):

            p.setJointMotorControl2(
                robot,
                joint_id,
                p.TORQUE_CONTROL,
                force=float(torque[i]),
            )

        p.stepSimulation()

        if step % 240 == 0:

            base_pos, base_orn = p.getBasePositionAndOrientation(robot)

            roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)

            error = np.abs(STANDING_POSE - q)

            print(
                f"t={step / 240.0:5.1f}s | "
                f"z={base_pos[2]:.3f} | "
                f"roll={roll:+.3f} | "
                f"pitch={pitch:+.3f} | "
                f"mean_err={np.mean(error):.4f} | "
                f"max_err={np.max(error):.4f}"
            )

    # Read final state.
    joint_states = [
        p.getJointState(robot, joint_id)
        for joint_id in revolute_joints
    ]

    q = np.array(
        [s[0] for s in joint_states],
        dtype=np.float32,
    )

    qd = np.array(
        [s[1] for s in joint_states],
        dtype=np.float32,
    )

    base_pos, base_orn = p.getBasePositionAndOrientation(robot)
    roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)

    print()
    print("=" * 70)
    print("FINAL EQUILIBRIUM")
    print("=" * 70)

    print(f"Base position : {base_pos}")
    print(f"Roll          : {roll:+.5f} rad")
    print(f"Pitch         : {pitch:+.5f} rad")
    print(f"Yaw           : {yaw:+.5f} rad")

    print()
    print("Joint configuration:")

    for i, angle in enumerate(q):
        print(
            f"{i:2d}: "
            f"nominal={STANDING_POSE[i]:+.4f} | "
            f"actual={angle:+.4f} | "
            f"error={STANDING_POSE[i] - angle:+.4f}"
        )

    print()
    print("Mean joint error:", np.mean(np.abs(STANDING_POSE - q)))
    print("Max joint error :", np.max(np.abs(STANDING_POSE - q)))

    print()
    print("Keep this configuration as a diagnostic result.")
    print("Do NOT train PPO yet.")

    time.sleep(2)
    p.disconnect()


if __name__ == "__main__":
    main()
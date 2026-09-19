import numpy as np


class PDController:
    """
    Explicit joint-level PD controller.

    PPO provides desired joint positions.
    The PD controller converts desired positions
    into joint torques.

    tau = Kp * (q_des - q) - Kd * q_dot
    """

    def __init__(
        self,
        kp=40.0,
        kd=1.0,
        torque_limit=33.5,
    ):
        self.kp = float(kp)
        self.kd = float(kd)
        self.torque_limit = float(torque_limit)

    def compute_torque(
        self,
        q_des,
        q,
        q_dot,
    ):
        q_des = np.asarray(q_des, dtype=np.float32)
        q = np.asarray(q, dtype=np.float32)
        q_dot = np.asarray(q_dot, dtype=np.float32)

        position_error = q_des - q

        torque = (
            self.kp * position_error
            - self.kd * q_dot
        )

        torque = np.clip(
            torque,
            -self.torque_limit,
            self.torque_limit,
        )

        return torque.astype(np.float32)
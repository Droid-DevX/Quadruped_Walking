"""
Task 1 — A1 PPO Evaluation + Rendering

Run from project root:
    python evaluation\task1_evaluation.py

Edit the CONFIG section below when you want to change the model,
normalization file, number of episodes, steps, or rendering.
"""

from pathlib import Path
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environments.env_flat_terrain import QuadrupedEnv


# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "task1_flat" / "ppo_a1_task1_smooth_final.zip"
NORM_PATH = PROJECT_ROOT / "checkpoints" / "task1_flat" / "ppo_a1_task1_smooth_final_vecnormalize.pkl"

N_EPISODES = 5
MAX_STEPS = 1000
TARGET_VELOCITY = 0.50
RENDER = True

# Real-time GUI playback.
# The environment itself should also use its validated physics/control loop.
REALTIME_RENDER = True

# -------------------- Visualisation --------------------
# Free camera is the default. Set CAMERA_FOLLOW=True only if automatic
# camera tracking is wanted.
CAMERA_FOLLOW = False
CAMERA_DISTANCE = 2.8
CAMERA_YAW = 135.0
CAMERA_PITCH = -20.0
CAMERA_LOOK_AHEAD = 0.65
CAMERA_HEIGHT_OFFSET = 0.10
CAMERA_UPDATE_EVERY = 4

SHOW_TRAJECTORY = True
SHOW_STATUS_TEXT = True
TRAJECTORY_WIDTH = 2.5
EPISODE_PAUSE = 1.0

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results" / "task1"


def get_base_env(vec_env):
    return vec_env.venv.envs[0].unwrapped


def get_client_id(base_env):
    for name in ("client", "client_id", "physics_client", "physicsClientId"):
        value = getattr(base_env, name, None)
        if isinstance(value, (int, np.integer)):
            return int(value)

    info = p.getConnectionInfo()
    if info.get("isConnected", 0):
        return int(info.get("clientIndex", 0))
    return 0


def read_robot_state(base_env):
    client = get_client_id(base_env)
    robot_id = int(base_env.robot_id)

    pos, orn = p.getBasePositionAndOrientation(
        robot_id, physicsClientId=client
    )
    lin_vel, ang_vel = p.getBaseVelocity(
        robot_id, physicsClientId=client
    )
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)

    joint_indices = list(base_env.joint_indices)
    states = [
        p.getJointState(robot_id, j, physicsClientId=client)
        for j in joint_indices
    ]

    q = np.asarray([s[0] for s in states], dtype=np.float32)
    qd = np.asarray([s[1] for s in states], dtype=np.float32)

    q_des = np.asarray(
        getattr(base_env, "current_q_des", np.zeros(len(q))),
        dtype=np.float32,
    )

    tau_value = getattr(base_env, "last_torque", None)
    if tau_value is None:
        tau_value = getattr(base_env, "_last_torque", None)

    if tau_value is not None:
        tau = np.asarray(tau_value, dtype=np.float32)
    else:
        controller = getattr(base_env, "controller", None)
        kp = float(getattr(controller, "kp", 40.0))
        kd = float(getattr(controller, "kd", 1.0))
        limit = float(getattr(controller, "torque_limit", 33.5))
        tau = np.clip(kp * (q_des - q) - kd * qd, -limit, limit)

    foot_links = []
    for i in range(p.getNumJoints(robot_id, physicsClientId=client)):
        info = p.getJointInfo(robot_id, i, physicsClientId=client)
        name = info[12].decode("utf-8", errors="ignore").lower()
        if "foot" in name or "toe" in name:
            foot_links.append(i)

    if len(foot_links) < 4:
        foot_links = list(getattr(base_env, "foot_link_indices", []))

    contacts = []
    for link in foot_links[:4]:
        points = p.getContactPoints(
            bodyA=robot_id,
            linkIndexA=link,
            physicsClientId=client,
        )
        contacts.append(float(bool(points)))

    while len(contacts) < 4:
        contacts.append(0.0)

    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2]),
        "vx": float(lin_vel[0]),
        "vy": float(lin_vel[1]),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
        "q": q,
        "qd": qd,
        "q_des": q_des,
        "tau": tau,
        "contacts": np.asarray(contacts[:4], dtype=np.float32),
    }


def setup_visualization(base_env):
    """Configure PyBullet visuals while leaving the free camera untouched."""
    if not RENDER:
        return

    client = get_client_id(base_env)
    try:
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, 1, physicsClientId=client
        )
        # Keep PyBullet's native GUI and camera controls visible.
        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI, 1, physicsClientId=client
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_MOUSE_PICKING, 1, physicsClientId=client
        )
    except Exception:
        pass

    # Deliberately no resetDebugVisualizerCamera() here.
    # The user owns the camera.

def update_visualization(base_env, state, previous_xy, step, text_id=-1):
    """Update camera, trajectory and compact status overlay."""
    if not RENDER:
        return text_id, (state["x"], state["y"])

    client = get_client_id(base_env)
    x, y, z = state["x"], state["y"], state["z"]

    # Optional follow mode. OFF by default so mouse rotation, pan and zoom
    # remain completely free.
    if CAMERA_FOLLOW and step % CAMERA_UPDATE_EVERY == 0:
        target = [x + CAMERA_LOOK_AHEAD, y, z + CAMERA_HEIGHT_OFFSET]
        p.resetDebugVisualizerCamera(
            cameraDistance=CAMERA_DISTANCE,
            cameraYaw=CAMERA_YAW,
            cameraPitch=CAMERA_PITCH,
            cameraTargetPosition=target,
            physicsClientId=client,
        )

    # Leave a persistent ground-level trajectory behind the robot.
    if SHOW_TRAJECTORY and previous_xy is not None:
        p.addUserDebugLine(
            [previous_xy[0], previous_xy[1], 0.012],
            [x, y, 0.012],
            lineColorRGB=[0.10, 0.75, 1.00],
            lineWidth=TRAJECTORY_WIDTH,
            lifeTime=0,
            physicsClientId=client,
        )

    if SHOW_STATUS_TEXT:
        text = (
            f"A1 PPO  |  Vx: {state['vx']:.2f} m/s  |  "
            f"Target: {TARGET_VELOCITY:.2f}  |  "
            f"Height: {z:.3f} m  |  "
            f"Roll: {np.degrees(state['roll']):+.1f}°  "
            f"Pitch: {np.degrees(state['pitch']):+.1f}°"
        )
        text_pos = [x, y, z + 0.45]
        text_id = p.addUserDebugText(
            text,
            text_pos,
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.25,
            lifeTime=0.10,
            replaceItemUniqueId=text_id,
            physicsClientId=client,
        )

    return text_id, (x, y)



def update_keyboard_camera(base_env):
    """Interactive PyBullet camera control. Click GUI first for keyboard focus."""
    if not RENDER:
        return
    client=get_client_id(base_env)
    try:
        events=p.getKeyboardEvents(physicsClientId=client)
        cam=p.getDebugVisualizerCamera(physicsClientId=client)
        distance=float(cam[10]); yaw=float(cam[8]); pitch=float(cam[9])
        target=np.array(cam[11],dtype=np.float64)
        def held(key):
            return bool(events.get(key,0) & p.KEY_IS_DOWN)
        move_speed=max(0.015,distance*0.02); zoom_speed=max(0.025,distance*0.05)
        yr=np.radians(yaw)
        forward=np.array([np.cos(yr),np.sin(yr),0.0])
        right=np.array([-np.sin(yr),np.cos(yr),0.0])
        changed=False
        # if held(ord('w')): target+=forward*move_speed; changed=True
        # if held(ord('s')): target-=forward*move_speed; changed=True
        # if held(ord('a')): target-=right*move_speed; changed=True
        # if held(ord('d')): target+=right*move_speed; changed=True
        # if held(p.B3G_LEFT_ARROW): yaw-=2.5; changed=True
        # if held(p.B3G_RIGHT_ARROW): yaw+=2.5; changed=True
        # if held(p.B3G_UP_ARROW): target+=forward*move_speed; changed=True
        # if held(p.B3G_DOWN_ARROW): target-=forward*move_speed; changed=True
        if held(ord('q')): distance+=zoom_speed; changed=True
        if held(ord('e')): distance-=zoom_speed; changed=True
        reset=held(ord('r'))
        if reset and not getattr(base_env,'_camera_reset_down',False):
            distance=CAMERA_DISTANCE; yaw=CAMERA_YAW; pitch=CAMERA_PITCH
            try:
                pos,_=p.getBasePositionAndOrientation(int(base_env.robot_id),physicsClientId=client)
                target=np.array([pos[0]+CAMERA_LOOK_AHEAD,pos[1],pos[2]+CAMERA_HEIGHT_OFFSET])
            except Exception: pass
            changed=True
        base_env._camera_reset_down=reset
        if changed:
            p.resetDebugVisualizerCamera(cameraDistance=float(np.clip(distance,0.3,20.0)),cameraYaw=yaw,cameraPitch=float(np.clip(pitch,-89,89)),cameraTargetPosition=target.tolist(),physicsClientId=client)
    except Exception:
        pass

def realtime_wait(next_frame_time):
    """Synchronize the 60 Hz control loop to wall-clock time."""
    if not (RENDER and REALTIME_RENDER):
        return next_frame_time

    period = 1.0 / 60.0
    next_frame_time += period
    remaining = next_frame_time - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
    else:
        # If the machine falls behind, resync instead of accumulating lag.
        next_frame_time = time.perf_counter()
    return next_frame_time

def run_episode(model, env):
    obs = env.reset()
    base_env = get_base_env(env)

    state0 = read_robot_state(base_env)
    initial_x = state0["x"]
    initial_y = state0["y"]

    setup_visualization(base_env)
    previous_xy = (state0["x"], state0["y"])
    text_id = -1
    next_frame_time = time.perf_counter()

    data = {
        "time": [], "x": [], "y": [], "vx": [], "vy": [], "z": [],
        "roll": [], "pitch": [], "yaw": [],
        "joint_pos": [], "joint_des": [], "joint_vel": [],
        "torque": [], "contacts": [], "reward": [],
    }

    total_reward = 0.0
    ended = False

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)

        state = read_robot_state(get_base_env(env))

        data["time"].append((step + 1) / 60.0)
        for key in ("x", "y", "vx", "vy", "z", "roll", "pitch", "yaw"):
            data[key].append(state[key])

        data["joint_pos"].append(state["q"])
        data["joint_des"].append(state["q_des"])
        data["joint_vel"].append(state["qd"])
        data["torque"].append(state["tau"])
        data["contacts"].append(state["contacts"])
        data["reward"].append(float(reward[0]))

        total_reward += float(reward[0])

        text_id, previous_xy = update_visualization(
            get_base_env(env), state, previous_xy, step, text_id
        )
        update_keyboard_camera(get_base_env(env))
        next_frame_time = realtime_wait(next_frame_time)

        if bool(dones[0]):
            ended = True
            break

    for key in [
        "time", "x", "y", "vx", "vy", "z",
        "roll", "pitch", "yaw", "reward"
    ]:
        data[key] = np.asarray(data[key], dtype=np.float32)

    for key in ["joint_pos", "joint_des", "joint_vel", "torque", "contacts"]:
        data[key] = np.asarray(data[key], dtype=np.float32)

    if len(data["x"]):
        dx = np.diff(np.r_[initial_x, data["x"]])
        dy = np.diff(np.r_[initial_y, data["y"]])
        forward_displacement = float(data["x"][-1] - initial_x)
        lateral_drift = float(data["y"][-1] - initial_y)
        path_length = float(np.sum(np.sqrt(dx * dx + dy * dy)))
        velocity_distance = float(np.sum(data["vx"]) / 60.0)
    else:
        forward_displacement = lateral_drift = path_length = velocity_distance = 0.0

    metrics = {
        "steps": len(data["time"]),
        "duration": len(data["time"]) / 60.0,
        "reward": total_reward,
        "forward_displacement": forward_displacement,
        "path_length": path_length,
        "lateral_drift": lateral_drift,
        "mean_vx": float(np.mean(data["vx"])) if len(data["vx"]) else 0.0,
        "mean_vy": float(np.mean(data["vy"])) if len(data["vy"]) else 0.0,
        "velocity_distance": velocity_distance,
        "max_roll": float(np.max(np.abs(data["roll"]))) if len(data["roll"]) else 0.0,
        "max_pitch": float(np.max(np.abs(data["pitch"]))) if len(data["pitch"]) else 0.0,
        "min_z": float(np.min(data["z"])) if len(data["z"]) else 0.0,
        "max_torque": float(np.max(np.abs(data["torque"]))) if len(data["torque"]) else 0.0,
        "mean_torque": float(np.mean(np.abs(data["torque"]))) if len(data["torque"]) else 0.0,
        "mean_contacts": float(np.mean(np.sum(data["contacts"], axis=1))) if len(data["contacts"]) else 0.0,
        "completed": (not ended) and len(data["time"]) >= MAX_STEPS,
    }

    return data, metrics


def save_plots(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t = data["time"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / name, dpi=150)
        plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(t, data["vx"])
    plt.axhline(TARGET_VELOCITY, linestyle="--", label=f"target {TARGET_VELOCITY:.2f} m/s")
    plt.xlabel("Time (s)")
    plt.ylabel("Forward velocity vx (m/s)")
    plt.title("Task 1 — Forward Velocity")
    plt.legend()
    plt.grid(True)
    save(fig, "velocity.png")

    fig = plt.figure(figsize=(10, 5))
    plt.plot(t, data["x"])
    plt.xlabel("Time (s)")
    plt.ylabel("Base X position (m)")
    plt.title("Task 1 — Forward Displacement")
    plt.grid(True)
    save(fig, "displacement.png")

    fig = plt.figure(figsize=(10, 5))
    plt.plot(t, np.degrees(data["roll"]), label="roll")
    plt.plot(t, np.degrees(data["pitch"]), label="pitch")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Task 1 — Body Orientation")
    plt.legend()
    plt.grid(True)
    save(fig, "orientation.png")

    fig = plt.figure(figsize=(10, 5))
    plt.plot(t, data["z"])
    plt.xlabel("Time (s)")
    plt.ylabel("Base height (m)")
    plt.title("Task 1 — Base Height")
    plt.grid(True)
    save(fig, "height.png")

    fig = plt.figure(figsize=(10, 5))
    plt.plot(t, np.max(np.abs(data["torque"]), axis=1))
    plt.axhline(33.5, linestyle="--", label="33.5 Nm limit")
    plt.xlabel("Time (s)")
    plt.ylabel("Maximum |torque| (Nm)")
    plt.title("Task 1 — Maximum Joint Torque")
    plt.legend()
    plt.grid(True)
    save(fig, "max_torque.png")

    fig = plt.figure(figsize=(10, 4))
    plt.imshow(
        data["contacts"].T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=[t[0], t[-1], 0, 4],
    )
    plt.yticks([0.5, 1.5, 2.5, 3.5], ["FR", "FL", "RR", "RL"])
    plt.xlabel("Time (s)")
    plt.ylabel("Foot")
    plt.title("Task 1 — Foot Contacts")
    save(fig, "foot_contacts.png")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not NORM_PATH.exists():
        raise FileNotFoundError(f"Normalization file not found: {NORM_PATH}")

    print("=" * 72)
    print("CAMERA: click PyBullet window | WASD move | Arrows rotate/move | Q/E zoom | R reset")
    print("TASK 1 — PPO EVALUATION + RENDERING")
    print("=" * 72)
    print(f"Model      : {MODEL_PATH}")
    print(f"VecNorm    : {NORM_PATH}")
    print(f"Episodes   : {N_EPISODES}")
    print(f"Steps      : {MAX_STEPS}")
    print(f"Target Vx  : {TARGET_VELOCITY:.2f} m/s")
    print(f"Render     : {RENDER}")
    print()

    vec_env = DummyVecEnv([
        lambda: QuadrupedEnv(
            terrain_id=0,
            render=RENDER,
            difficulty=0.0,
            target_velocity=TARGET_VELOCITY,
            max_episode_steps=MAX_STEPS,
        )
    ])

    env = VecNormalize.load(str(NORM_PATH), vec_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(MODEL_PATH), env=env, device="cpu")

    results = []

    try:
        for episode in range(1, N_EPISODES + 1):
            data, metrics = run_episode(model, env)
            results.append(metrics)

            print("-" * 72)
            print(f"Episode {episode}/{N_EPISODES}")
            print(f"Steps                         : {metrics['steps']}")
            print(f"Duration                      : {metrics['duration']:.3f} s")
            print(f"Reward                        : {metrics['reward']:.2f}")
            print(f"Initial/Final X displacement  : {metrics['forward_displacement']:.4f} m")
            print(f"Path length (XY)              : {metrics['path_length']:.4f} m")
            print(f"Lateral drift                 : {metrics['lateral_drift']:.4f} m")
            print(f"Mean Vx                       : {metrics['mean_vx']:.4f} m/s")
            print(f"Mean Vy                       : {metrics['mean_vy']:.4f} m/s")
            print(f"Velocity-integrated distance  : {metrics['velocity_distance']:.4f} m")
            print(f"Max |roll|                    : {np.degrees(metrics['max_roll']):.2f} deg")
            print(f"Max |pitch|                   : {np.degrees(metrics['max_pitch']):.2f} deg")
            print(f"Minimum base height           : {metrics['min_z']:.4f} m")
            print(f"Maximum torque                : {metrics['max_torque']:.4f} Nm")
            print(f"Mean feet in contact          : {metrics['mean_contacts']:.2f}")
            print(f"Completed full episode        : {metrics['completed']}")

            if episode == 1:
                save_plots(data)

            if RENDER and episode < N_EPISODES:
                time.sleep(EPISODE_PAUSE)

        print("\n" + "=" * 72)
        print("TASK 1 — TRAJECTORY METRIC SUMMARY")
        print("=" * 72)

        def summary(key):
            values = [r[key] for r in results]
            return np.mean(values), np.std(values)

        for key, label, unit in [
            ("forward_displacement", "Forward displacement", "m"),
            ("path_length", "Mean XY path length", "m"),
            ("lateral_drift", "Lateral drift", "m"),
            ("mean_vx", "Mean Vx", "m/s"),
            ("velocity_distance", "Velocity-integrated distance", "m"),
            ("max_roll", "Max |roll|", "deg"),
            ("max_pitch", "Max |pitch|", "deg"),
            ("min_z", "Minimum base height", "m"),
            ("max_torque", "Maximum torque", "Nm"),
            ("mean_contacts", "Mean feet in contact", ""),
        ]:
            mean, std = summary(key)
            if "roll" in key or "pitch" in key:
                mean, std = np.degrees(mean), np.degrees(std)
            print(f"{label:30s}: {mean:.4f} ± {std:.4f} {unit}")

        mean, std = summary("reward")
        print(f"{'Mean reward':30s}: {mean:.2f} ± {std:.2f}")
        print(f"{'Full episodes':30s}: {sum(r['completed'] for r in results)}/{len(results)}")
        print(f"\nPlots saved to: {OUTPUT_DIR}")

    finally:
        env.close()


if __name__ == "__main__":
    main()

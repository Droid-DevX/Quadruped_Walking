"""
Task 2 — PPO trajectory-metric evaluator

Purpose:
    Evaluate the trained Task-2 PPO + PD policy using the SAME
    VecNormalize statistics used during training and report physical
    trajectory metrics from the raw PyBullet environment.

Architecture:
    PPO -> normalized observation -> action
        -> rate-limited desired joint positions
        -> explicit PD -> torque -> PyBullet A1

IMPORTANT:
    This evaluator does NOT estimate distance from reward or from a
    non-existent "x_position" info key.

    The environment exposes:
        info["base_x"]
        info["base_y"]
        info["base_velocity_x"]
        info["base_velocity_y"]
        info["base_height"]
        info["roll"]
        info["pitch"]
        info["max_torque"]

    Therefore:
        forward displacement = final base_x - initial base_x
        lateral drift        = final base_y - initial base_y

    PPO receives ONLY VecNormalize-normalized observations.
    Physical metrics are read ONLY from the underlying raw environment.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environments.env_uneven_terrain import QuadrupedTerrainEnv


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "task2_terrain"
    / "ppo_a1_task2_terrain_final.zip"
)

VECNORM_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "task2_terrain"
    / "ppo_a1_task2_terrain_final_vecnormalize.pkl"
)

DIFFICULTY = 1.00
TARGET_VELOCITY = 0.50
MAX_STEPS = 1000
N_EPISODES = 5

# Keep visualization OFF for this metric-validation pass.
# We will handle rendering separately after the metrics are verified.
# True = PyBullet GUI, False = headless evaluation
RENDER = True
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


# ---------------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------------

def make_env():
    def _make():
        return QuadrupedTerrainEnv(
            difficulty=DIFFICULTY,
            target_velocity=TARGET_VELOCITY,
            max_episode_steps=MAX_STEPS,
            render=RENDER,
        )

    return _make


def get_client_id(base_env):
    for name in ("client", "client_id", "physics_client", "physicsClientId"):
        value = getattr(base_env, name, None)
        if isinstance(value, (int, np.integer)):
            return int(value)
    info = p.getConnectionInfo()
    return int(info.get("clientIndex", 0)) if info.get("isConnected", 0) else 0


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

def update_visualization(base_env, info, previous_xy, step, text_id=-1):
    if not RENDER:
        return text_id, previous_xy

    client = get_client_id(base_env)
    x = float(info["base_x"])
    y = float(info["base_y"])
    z = float(info["base_height"])

    # Optional follow mode. OFF by default so mouse rotation, pan and zoom
    # remain completely free.
    if CAMERA_FOLLOW and step % CAMERA_UPDATE_EVERY == 0:
        p.resetDebugVisualizerCamera(
            cameraDistance=CAMERA_DISTANCE,
            cameraYaw=CAMERA_YAW,
            cameraPitch=CAMERA_PITCH,
            cameraTargetPosition=[x + CAMERA_LOOK_AHEAD, y, z + CAMERA_HEIGHT_OFFSET],
            physicsClientId=client,
        )

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
            f"A1 PPO | Difficulty: {DIFFICULTY:.2f} | "
            f"Vx: {float(info['base_velocity_x']):.2f} m/s | "
            f"Target: {TARGET_VELOCITY:.2f} | "
            f"Height: {z:.3f} m | "
            f"Roll: {np.degrees(float(info['roll'])):+.1f}° "
            f"Pitch: {np.degrees(float(info['pitch'])):+.1f}°"
        )
        text_id = p.addUserDebugText(
            text,
            [x, y, z + 0.45],
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
    if not (RENDER and REALTIME_RENDER):
        return next_frame_time
    period = 1.0 / 60.0
    next_frame_time += period
    remaining = next_frame_time - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
    else:
        next_frame_time = time.perf_counter()
    return next_frame_time

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

def evaluate_episode(model, vec_env, raw_env, episode_number):
    """
    Run one deterministic episode.

    Returns physical trajectory metrics measured directly from the raw
    PyBullet environment info dictionary.
    """

    obs = vec_env.reset()

    # IMPORTANT:
    # VecNormalize is used for the policy input.
    # The raw environment is used only for physical diagnostics.
    base_env = raw_env.envs[0].unwrapped
    initial_info = base_env._get_info()
    setup_visualization(base_env)
    previous_xy = (float(initial_info["base_x"]), float(initial_info["base_y"]))
    text_id = -1
    next_frame_time = time.perf_counter()

    x_positions = [float(initial_info["base_x"])]
    y_positions = [float(initial_info["base_y"])]
    vx_values = []
    vy_values = []
    heights = []
    rolls = []
    pitches = []
    torques = []
    contacts = []

    episode_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, dones, infos = vec_env.step(action)

        # DummyVecEnv returns one environment.
        info = infos[0]

        episode_reward += float(reward[0])
        steps += 1

        # Physical state comes from the raw environment info.
        x = float(info["base_x"])
        y = float(info["base_y"])
        vx = float(info["base_velocity_x"])
        vy = float(info["base_velocity_y"])
        z = float(info["base_height"])
        roll = float(info["roll"])
        pitch = float(info["pitch"])
        torque = float(info["max_torque"])

        x_positions.append(x)
        y_positions.append(y)
        vx_values.append(vx)
        vy_values.append(vy)
        heights.append(z)
        rolls.append(roll)
        pitches.append(pitch)
        torques.append(torque)

        # Physical foot contacts are not included in info, so obtain them
        # directly from the raw environment for diagnostics.
        raw = raw_env.envs[0].unwrapped
        if hasattr(raw, "_get_foot_contacts"):
            foot_contacts = raw._get_foot_contacts()
            contacts.append(float(np.sum(foot_contacts)))

        text_id, previous_xy = update_visualization(
            base_env, info, previous_xy, steps, text_id
        )
        update_keyboard_camera(base_env)
        next_frame_time = realtime_wait(next_frame_time)

        terminated = bool(dones[0])

        # Gymnasium's termination/truncation distinction is not preserved
        # separately by the SB3 VecEnv API. The environment's max-step limit
        # is therefore used as the truncation criterion here.
        truncated = steps >= MAX_STEPS

    # -----------------------------------------------------------------------
    # CORRECT TRAJECTORY METRICS
    # -----------------------------------------------------------------------

    initial_x = x_positions[0]
    final_x = x_positions[-1]

    initial_y = y_positions[0]
    final_y = y_positions[-1]

    # Net physical displacement.
    forward_displacement = final_x - initial_x
    lateral_drift = final_y - initial_y

    duration = steps / 60.0

    mean_vx = float(np.mean(vx_values)) if vx_values else 0.0
    mean_vy = float(np.mean(vy_values)) if vy_values else 0.0

    # Independent sanity-check estimate.
    velocity_integrated_distance = float(np.sum(vx_values) / 60.0)

    # Total XY path length.
    path_length = 0.0
    if len(x_positions) > 1:
        dx = np.diff(np.asarray(x_positions))
        dy = np.diff(np.asarray(y_positions))
        path_length = float(np.sum(np.sqrt(dx * dx + dy * dy)))

    max_abs_roll = (
        float(np.max(np.abs(rolls))) if rolls else 0.0
    )
    max_abs_pitch = (
        float(np.max(np.abs(pitches))) if pitches else 0.0
    )
    min_height = (
        float(np.min(heights)) if heights else 0.0
    )
    max_torque = (
        float(np.max(torques)) if torques else 0.0
    )
    mean_contacts = (
        float(np.mean(contacts)) if contacts else 0.0
    )

    # The evaluator should never silently accept a physically inconsistent
    # result. This warning is deliberately informational rather than a hard
    # failure because small differences are expected from integration.
    if abs(forward_displacement) < 0.5 and abs(mean_vx) > 0.3:
        print(
            "\nWARNING: forward displacement is unusually small compared "
            "with measured mean Vx."
        )
        print(
            f"         displacement={forward_displacement:.3f} m, "
            f"mean_vx={mean_vx:.3f} m/s"
        )

    print(
        f"\nEpisode {episode_number}/{N_EPISODES}"
    )
    print("-" * 72)
    print(f"Steps                         : {steps}")
    print(f"Duration                      : {duration:.3f} s")
    print(f"Reward                        : {episode_reward:.2f}")
    print(f"Initial base X                : {initial_x:.4f} m")
    print(f"Final base X                  : {final_x:.4f} m")
    print(f"Forward displacement          : {forward_displacement:.4f} m")
    print(f"Path length (XY)              : {path_length:.4f} m")
    print(f"Lateral drift                 : {lateral_drift:.4f} m")
    print(f"Mean Vx                       : {mean_vx:.4f} m/s")
    print(f"Mean Vy                       : {mean_vy:.4f} m/s")
    print(
        f"Velocity-integrated distance  : "
        f"{velocity_integrated_distance:.4f} m"
    )
    print(
        f"Max |roll|                    : "
        f"{np.degrees(max_abs_roll):.2f} deg"
    )
    print(
        f"Max |pitch|                   : "
        f"{np.degrees(max_abs_pitch):.2f} deg"
    )
    print(f"Minimum base height           : {min_height:.4f} m")
    print(f"Maximum torque                : {max_torque:.4f} Nm")
    print(f"Mean feet in contact          : {mean_contacts:.2f}")
    print(f"Completed full episode        : {steps >= MAX_STEPS}")

    return {
        "steps": steps,
        "duration": duration,
        "reward": episode_reward,
        "forward_displacement": forward_displacement,
        "path_length": path_length,
        "lateral_drift": lateral_drift,
        "mean_vx": mean_vx,
        "mean_vy": mean_vy,
        "velocity_integrated_distance": velocity_integrated_distance,
        "max_roll_deg": np.degrees(max_abs_roll),
        "max_pitch_deg": np.degrees(max_abs_pitch),
        "min_height": min_height,
        "max_torque": max_torque,
        "mean_contacts": mean_contacts,
        "completed": steps >= MAX_STEPS,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("CAMERA: click PyBullet window | WASD move | Arrows rotate/move | Q/E zoom | R reset")
    print("TASK 2 — PPO EVALUATION + RENDERING")
    print("=" * 72)
    print(f"Model      : {MODEL_PATH}")
    print(f"VecNorm    : {VECNORM_PATH}")
    print(f"Difficulty : {DIFFICULTY:.2f}")
    print(f"Episodes   : {N_EPISODES}")
    print(f"Render     : {RENDER}")
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not VECNORM_PATH.exists():
        raise FileNotFoundError(
            f"VecNormalize file not found:\n{VECNORM_PATH}"
        )

    # The factory is intentionally called here.
    # Do NOT use DummyVecEnv([make_env]) because make_env() itself
    # returns the environment factory.
    raw_env = DummyVecEnv([make_env()])

    # CRITICAL:
    # Load the exact Task-2 normalization statistics used during training.
    vec_env = VecNormalize.load(
        str(VECNORM_PATH),
        raw_env,
    )

    vec_env.training = False
    vec_env.norm_reward = False
    vec_env.clip_obs = 10.0

    model = PPO.load(
        str(MODEL_PATH),
        env=vec_env,
        device="cpu",
    )

    results = []

    try:
        for episode in range(1, N_EPISODES + 1):
            results.append(
                evaluate_episode(
                    model=model,
                    vec_env=vec_env,
                    raw_env=raw_env,
                    episode_number=episode,
                )
            )
            if RENDER and episode < N_EPISODES:
                time.sleep(EPISODE_PAUSE)
    finally:
        vec_env.close()

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------

    def mean(key):
        return float(np.mean([r[key] for r in results]))

    def std(key):
        return float(np.std([r[key] for r in results]))

    print("\n" + "=" * 72)
    print("TASK 2 — TRAJECTORY METRIC SUMMARY")
    print("=" * 72)

    print(
        f"Difficulty                    : {DIFFICULTY:.2f}"
    )
    print(
        f"Mean steps                    : "
        f"{mean('steps'):.1f} ± {std('steps'):.1f}"
    )
    print(
        f"Mean reward                   : "
        f"{mean('reward'):.2f} ± {std('reward'):.2f}"
    )
    print(
        f"Forward displacement          : "
        f"{mean('forward_displacement'):.4f} ± "
        f"{std('forward_displacement'):.4f} m"
    )
    print(
        f"Mean XY path length           : "
        f"{mean('path_length'):.4f} ± "
        f"{std('path_length'):.4f} m"
    )
    print(
        f"Lateral drift                 : "
        f"{mean('lateral_drift'):.4f} ± "
        f"{std('lateral_drift'):.4f} m"
    )
    print(
        f"Mean Vx                       : "
        f"{mean('mean_vx'):.4f} ± {std('mean_vx'):.4f} m/s"
    )
    print(
        f"Velocity-integrated distance  : "
        f"{mean('velocity_integrated_distance'):.4f} ± "
        f"{std('velocity_integrated_distance'):.4f} m"
    )
    print(
        f"Max |roll|                    : "
        f"{mean('max_roll_deg'):.2f} ± {std('max_roll_deg'):.2f} deg"
    )
    print(
        f"Max |pitch|                   : "
        f"{mean('max_pitch_deg'):.2f} ± {std('max_pitch_deg'):.2f} deg"
    )
    print(
        f"Minimum base height           : "
        f"{mean('min_height'):.4f} ± {std('min_height'):.4f} m"
    )
    print(
        f"Maximum torque                : "
        f"{mean('max_torque'):.4f} ± {std('max_torque'):.4f} Nm"
    )
    print(
        f"Mean feet in contact          : "
        f"{mean('mean_contacts'):.2f} ± {std('mean_contacts'):.2f}"
    )
    print(
        f"Full episodes                 : "
        f"{sum(r['completed'] for r in results)}/{len(results)}"
    )

    print("\nMetric definitions:")
    print("  Forward displacement = final base_x - initial base_x")
    print("  Lateral drift        = final base_y - initial base_y")
    print("  Path length          = accumulated XY trajectory length")
    print("  Mean Vx              = mean physical PyBullet base velocity")
    print("  Velocity distance    = integral(Vx dt), independent sanity check")


if __name__ == "__main__":
    main()

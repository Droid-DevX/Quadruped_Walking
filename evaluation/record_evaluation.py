"""
Record PPO evaluations for Unitree A1 Task 1 and Task 2.

Run from project root:
    python evaluation\record_evaluation.py

The script intentionally:
- Uses the actual environment class names:
    Task 1 -> QuadrupedEnv
    Task 2 -> QuadrupedTerrainEnv
- Automatically finds the Task-1 final model/norm in either:
    checkpoints/task1_smooth/
    checkpoints/task1_flat/
- Uses the Task-2 final model/norm in:
    checkpoints/task2_terrain/
- Resets the environment BEFORE initializing the camera.
- Initializes the PyBullet camera only once per task.
- Never resets/follows the camera during an episode.
- Captures the current PyBullet camera, so manual rotate/zoom/pan is preserved.
- Adds a fixed video HUD at the top of the recorded frame.
"""

from pathlib import Path
import sys
import time
import numpy as np
import cv2
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "recordings"

EPISODE_STEPS = 1000
FPS = 60
TASK2_DIFFICULTY = 1.0
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

RENDER = True
# Normal-speed GUI playback for screen recording.
# The environment remains 60 Hz control / 240 Hz physics.
REALTIME = True
# IMPORTANT: keep this False when using Windows screen recording.
# getCameraImage() + OpenCV encoding is expensive and can make the GUI
# appear much slower than the original evaluate.py.
RECORD_VIDEO = False
GUI_START_DELAY = 2.0

# Initial camera view only. After initialization, the script
# never changes the camera again.
CAMERA_DISTANCE = 3.4
CAMERA_YAW = 125.0
CAMERA_PITCH = -18.0

# Camera follows the robot while preserving manual camera angle/zoom.
CAMERA_FOLLOW = True
CAMERA_LOOK_AHEAD = 0.80
CAMERA_HEIGHT_OFFSET = 0.05
CAMERA_UPDATE_EVERY = 1


# ============================================================
# IMPORT PROJECT ENVIRONMENTS
# ============================================================

sys.path.insert(0, str(ROOT))

from environments.env_flat_terrain import QuadrupedEnv
from environments.env_uneven_terrain import QuadrupedTerrainEnv


# ============================================================
# ARTIFACT RESOLUTION
# ============================================================

def first_existing(candidates, description):
    """Return the first existing path or raise a useful error."""
    for path in candidates:
        if path.exists():
            return path

    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(
        f"\n{description} not found.\n"
        f"Checked:\n{tried}\n\n"
        "Make sure the trained checkpoint is present in the project."
    )


def resolve_task1_artifacts():
    """
    Task-1 was saved under task1_flat in the training configuration,
    while an earlier recorder assumed task1_smooth. Support both so
    the recorder does not break because of the directory name.
    """
    model = first_existing(
        [
            ROOT / "checkpoints" / "task1_smooth" /
            "ppo_a1_task1_smooth_final.zip",
            ROOT / "checkpoints" / "task1_flat" /
            "ppo_a1_task1_smooth_final.zip",
            ROOT / "checkpoints" / "task1_flat" /
            "ppo_a1_task1_flat_final.zip",
        ],
        "Task 1 PPO model",
    )

    # Prefer a VecNormalize file with the same stem/directory.
    norm_candidates = [
        model.parent / "ppo_a1_task1_smooth_final_vecnormalize.pkl",
        model.parent / "ppo_a1_task1_flat_final_vecnormalize.pkl",
    ]

    norm = first_existing(
        norm_candidates,
        "Task 1 VecNormalize file",
    )

    return model, norm


def resolve_task2_artifacts():
    model = first_existing(
        [
            ROOT / "checkpoints" / "task2_terrain" /
            "ppo_a1_task2_terrain_final.zip",
        ],
        "Task 2 PPO model",
    )

    norm = first_existing(
        [
            ROOT / "checkpoints" / "task2_terrain" /
            "ppo_a1_task2_terrain_final_vecnormalize.pkl",
        ],
        "Task 2 VecNormalize file",
    )

    return model, norm


# ============================================================
# CAMERA
# ============================================================

def initialize_camera(robot_id, physics_client):
    """
    Initialize the camera once.

    After initialization, the camera follows the robot's position,
    while preserving the current distance/yaw/pitch. This means the
    user can still manually rotate and zoom the PyBullet camera while
    the camera target moves forward with the robot.
    """
    pos, _ = p.getBasePositionAndOrientation(
        robot_id,
        physicsClientId=physics_client,
    )

    p.resetDebugVisualizerCamera(
        cameraDistance=CAMERA_DISTANCE,
        cameraYaw=CAMERA_YAW,
        cameraPitch=CAMERA_PITCH,
        cameraTargetPosition=[
            float(pos[0]) + CAMERA_LOOK_AHEAD,
            float(pos[1]),
            float(pos[2]) + CAMERA_HEIGHT_OFFSET,
        ],
        physicsClientId=physics_client,
    )


def follow_camera(robot_id, physics_client):
    """
    Move the camera target with the robot without forcing a fixed view.

    The current PyBullet camera distance, yaw and pitch are read every
    update. Therefore manual mouse rotation/zoom is preserved.

    Only the target position follows the robot.
    """
    try:
        camera = p.getDebugVisualizerCamera(physicsClientId=physics_client)

        # PyBullet debug-camera fields.
        current_yaw = float(camera[8])
        current_pitch = float(camera[9])
        current_distance = float(camera[10])

        pos, _ = p.getBasePositionAndOrientation(
            robot_id,
            physicsClientId=physics_client,
        )

        target = [
            float(pos[0]) + CAMERA_LOOK_AHEAD,
            float(pos[1]),
            float(pos[2]) + CAMERA_HEIGHT_OFFSET,
        ]

        p.resetDebugVisualizerCamera(
            cameraDistance=current_distance,
            cameraYaw=current_yaw,
            cameraPitch=current_pitch,
            cameraTargetPosition=target,
            physicsClientId=physics_client,
        )
    except Exception:
        # Camera following should never stop the evaluation.
        pass


def get_camera_image():
    """
    Capture the current PyBullet debug camera and return
    a BGR frame suitable for OpenCV / VideoWriter.

    Works with PyBullet's returned RGB buffer whether it
    arrives as a flat 1-D array or already-shaped array.
    """

    width = VIDEO_WIDTH
    height = VIDEO_HEIGHT

    camera = p.getDebugVisualizerCamera()

    view_matrix = camera[2]
    projection_matrix = camera[3]

    img = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    rgba = np.asarray(img[2], dtype=np.uint8)

    # PyBullet can return the RGB/RGBA buffer as a flat array.
    expected_rgba = width * height * 4
    expected_rgb = width * height * 3

    if rgba.ndim == 1:

        if rgba.size == expected_rgba:
            rgba = rgba.reshape((height, width, 4))

        elif rgba.size == expected_rgb:
            rgba = rgba.reshape((height, width, 3))

        else:
            raise RuntimeError(
                f"Unexpected camera buffer size: {rgba.size}. "
                f"Expected {expected_rgba} (RGBA) or "
                f"{expected_rgb} (RGB)."
            )

    elif rgba.ndim == 2:
        # Handle unusual flattened 2-D representations.
        if rgba.size == expected_rgba:
            rgba = rgba.reshape((height, width, 4))

        elif rgba.size == expected_rgb:
            rgba = rgba.reshape((height, width, 3))

        else:
            raise RuntimeError(
                f"Unexpected 2-D camera buffer shape: {rgba.shape}"
            )

    elif rgba.ndim == 3:
        # Already correctly shaped.
        pass

    else:
        raise RuntimeError(
            f"Unexpected camera buffer dimensions: {rgba.ndim}"
        )

    # Convert RGB/RGBA -> BGR for OpenCV.
    if rgba.shape[2] == 4:
        frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    elif rgba.shape[2] == 3:
        frame = cv2.cvtColor(rgba, cv2.COLOR_RGB2BGR)

    else:
        raise RuntimeError(
            f"Unexpected number of image channels: {rgba.shape[2]}"
        )

    return frame


# ============================================================
# HUD
# ============================================================

def draw_text(frame, text, position, scale=0.70, thickness=2):
    """White text with a strong black outline."""
    x, y = position

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 4,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_hud(
    frame,
    task_name,
    difficulty,
    step,
    vx,
    target_v,
    height,
    roll_deg,
    pitch_deg,
):
    """Fixed HUD at the top of the recorded video."""
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (VIDEO_WIDTH, 96),
        (0, 0, 0),
        -1,
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0,
    )

    if difficulty is None:
        title = f"{task_name}  |  FLAT TERRAIN"
    else:
        title = (
            f"{task_name}  |  UNEVEN TERRAIN  |  "
            f"Difficulty: {difficulty:.2f}"
        )

    draw_text(
        frame,
        title,
        (25, 36),
        scale=0.78,
        thickness=2,
    )

    info = (
        f"Step: {step:04d}/{EPISODE_STEPS}    "
        f"Vx: {vx:+.3f} m/s    "
        f"Target: {target_v:.2f} m/s    "
        f"Height: {height:.3f} m    "
        f"Roll: {roll_deg:+.1f} deg    "
        f"Pitch: {pitch_deg:+.1f} deg"
    )

    draw_text(
        frame,
        info,
        (25, 75),
        scale=0.55,
        thickness=2,
    )

    return frame


def draw_bottom_banner(frame, text):
    """Final-frame message."""
    h, w = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, h - 85),
        (w, h),
        (0, 0, 0),
        -1,
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2

    size = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )[0]

    x = (w - size[0]) // 2
    y = h - 32

    draw_text(
        frame,
        text,
        (x, y),
        scale=scale,
        thickness=thickness,
    )

    return frame


# ============================================================
# ENVIRONMENT FACTORIES
# ============================================================

def make_task1_env():
    def _make():
        return QuadrupedEnv(
            terrain_id=0,
            render=RENDER,
            target_velocity=0.50,
            max_episode_steps=EPISODE_STEPS,
        )

    return _make


def make_task2_env():
    def _make():
        return QuadrupedTerrainEnv(
            render=RENDER,
            difficulty=TASK2_DIFFICULTY,
            target_velocity=0.50,
            max_episode_steps=EPISODE_STEPS,
        )

    return _make


# ============================================================
# PHYSICAL STATE
# ============================================================

def get_state(env):
    """
    Read the actual robot state directly from PyBullet.

    This works for both:
        Task 1 -> QuadrupedEnv
        Task 2 -> QuadrupedTerrainEnv

    It does not depend on task-specific _get_info() keys.
    """

    position, orientation = p.getBasePositionAndOrientation(
        env.robot_id,
        physicsClientId=env.physics_client,
    )

    linear_velocity, angular_velocity = p.getBaseVelocity(
        env.robot_id,
        physicsClientId=env.physics_client,
    )

    roll_rad, pitch_rad, _ = p.getEulerFromQuaternion(
        orientation
    )

    pos = np.asarray(position, dtype=np.float64)

    vx = float(linear_velocity[0])
    vy = float(linear_velocity[1])

    roll_deg = float(np.degrees(roll_rad))
    pitch_deg = float(np.degrees(pitch_rad))

    return (
        pos,
        vx,
        vy,
        roll_deg,
        pitch_deg,
    )

# ============================================================
# RECORD ONE TASK
# ============================================================

def record_task(
    task_name,
    model_path,
    norm_path,
    env_factory,
    difficulty,
    output_path,
):
    print("\n" + "=" * 72)
    print(f"RECORDING {task_name}")
    print("=" * 72)
    print(f"Model      : {model_path}")
    print(f"VecNorm    : {norm_path}")
    print(f"Output     : {output_path}")

    # Artifact checks happen before opening the GUI.
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not norm_path.exists():
        raise FileNotFoundError(
            f"VecNormalize file not found: {norm_path}"
        )

    vec_env = None
    writer = None

    try:
        # ----------------------------------------------------
        # Build raw environment
        # ----------------------------------------------------
        raw_vec_env = DummyVecEnv([env_factory])

        # ----------------------------------------------------
        # Load the EXACT normalization statistics used by PPO
        # ----------------------------------------------------
        vec_env = VecNormalize.load(
            str(norm_path),
            raw_vec_env,
        )

        vec_env.training = False
        vec_env.norm_reward = False

        # ----------------------------------------------------
        # Load PPO model
        # ----------------------------------------------------
        model = PPO.load(
            str(model_path),
            env=vec_env,
            device="cpu",
        )

        raw_env = vec_env.envs[0].unwrapped

        # ----------------------------------------------------
        # IMPORTANT: RESET BEFORE CAMERA INITIALIZATION
        # The robot does not exist until reset() creates it.
        # ----------------------------------------------------
        obs = vec_env.reset()

        # Match evaluate.py: use explicit environment stepping.
        # Do not let PyBullet run from wall-clock real-time simulation.
        p.setRealTimeSimulation(
            0,
            physicsClientId=raw_env.physics_client,
        )

        # Give PyBullet GUI time to open.
        time.sleep(GUI_START_DELAY)

        initialize_camera(
            raw_env.robot_id,
            raw_env.physics_client,
        )

        print()
        print("Camera initialized.")
        print("You can now manually rotate / zoom / pan the camera.")
        print("The camera follows the robot while preserving yaw/pitch/zoom.")
        print()

        # ----------------------------------------------------
        # Video writer
        # ----------------------------------------------------
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        writer = None

        if RECORD_VIDEO:
            writer = cv2.VideoWriter(
                str(output_path),
                FOURCC,
                FPS,
                (VIDEO_WIDTH, VIDEO_HEIGHT),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    "Could not open video writer."
                )

        last_frame = None
        last_vx = 0.0
        last_height = 0.0
        last_roll = 0.0
        last_pitch = 0.0

        # ----------------------------------------------------
        # Evaluation loop
        # ----------------------------------------------------
        # Match evaluate.py: one control step every 1/60 second.
        next_frame_time = time.perf_counter()

        for step in range(1, EPISODE_STEPS + 1):

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, rewards, dones, infos = vec_env.step(action)

            (
                pos,
                vx,
                vy,
                roll_deg,
                pitch_deg,
            ) = get_state(raw_env)

            # Keep the camera centered on the moving robot while preserving
            # the user's current yaw, pitch and zoom.
            if CAMERA_FOLLOW and step % CAMERA_UPDATE_EVERY == 0:
                follow_camera(
                    raw_env.robot_id,
                    raw_env.physics_client,
                )

            # ------------------------------------------------
            # Optional MP4 capture.
            #
            # When RECORD_VIDEO=False, do NOT call getCameraImage()
            # and do NOT encode a frame. This keeps the PyBullet GUI
            # timing essentially identical to evaluate.py, which is
            # what should be used for Windows screen recording.
            # ------------------------------------------------
            if RECORD_VIDEO:
                frame = get_camera_image()

                frame = draw_hud(
                    frame,
                    task_name=task_name,
                    difficulty=difficulty,
                    step=step,
                    vx=vx,
                    target_v=0.50,
                    height=pos[2],
                    roll_deg=roll_deg,
                    pitch_deg=pitch_deg,
                )

                writer.write(frame)

                last_frame = frame
                last_vx = vx
                last_height = float(pos[2])
                last_roll = roll_deg
                last_pitch = pitch_deg

            # ------------------------------------------------
            # IMPORTANT:
            # Use the SAME 60 Hz wall-clock pacing as evaluate.py.
            # The camera-follow operation does not alter physics.
            # ------------------------------------------------
            if REALTIME:
                next_frame_time += 1.0 / FPS
                remaining = next_frame_time - time.perf_counter()

                if remaining > 0:
                    time.sleep(remaining)
                else:
                    # Resynchronize if one iteration took too long.
                    next_frame_time = time.perf_counter()

            if dones[0]:
                print(
                    f"Episode terminated early at step {step}."
                )
                break

        # ----------------------------------------------------
        # Hold final frame for two seconds only when making MP4.
        # ----------------------------------------------------
        if RECORD_VIDEO and last_frame is not None:
            final_frame = get_camera_image()

            final_frame = draw_hud(
                final_frame,
                task_name=task_name,
                difficulty=difficulty,
                step=step,
                vx=last_vx,
                target_v=0.50,
                height=last_height,
                roll_deg=last_roll,
                pitch_deg=last_pitch,
            )

            final_frame = draw_bottom_banner(
                final_frame,
                f"{task_name} evaluation complete",
            )

            for _ in range(FPS * 2):
                writer.write(final_frame)

        if RECORD_VIDEO:
            print(f"Saved: {output_path}")
        else:
            print("Screen-recording mode complete. No MP4 encoding was performed.")

    finally:
        if writer is not None:
            writer.release()

        if vec_env is not None:
            vec_env.close()


# ============================================================
# COMBINE VIDEOS
# ============================================================

def combine_videos(clips, output_path):
    print("\nCreating combined evaluation video...")

    writer = cv2.VideoWriter(
        str(output_path),
        FOURCC,
        FPS,
        (VIDEO_WIDTH, VIDEO_HEIGHT),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create combined video: {output_path}"
        )

    try:
        for clip in clips:
            if not clip.exists():
                raise FileNotFoundError(
                    f"Expected recorded clip does not exist: {clip}"
                )

            cap = cv2.VideoCapture(str(clip))

            try:
                while True:
                    ok, frame = cap.read()

                    if not ok:
                        break

                    writer.write(frame)
            finally:
                cap.release()

    finally:
        writer.release()

    print(f"Combined video saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Resolve artifacts BEFORE launching any environment.
    task1_model, task1_norm = resolve_task1_artifacts()
    task2_model, task2_norm = resolve_task2_artifacts()

    task1_output = OUTPUT_DIR / "task1_flat.mp4"
    task2_output = OUTPUT_DIR / "task2_uneven.mp4"
    combined_output = OUTPUT_DIR / "quadruped_evaluation.mp4"

    print("\n" + "=" * 72)
    print("UNITREE A1 — PPO EVALUATION VIDEO RECORDER")
    print("=" * 72)
    print(f"Project    : {ROOT}")
    print(f"Resolution : {VIDEO_WIDTH}x{VIDEO_HEIGHT}")
    print(f"FPS        : {FPS}")
    print(f"Steps/task : {EPISODE_STEPS}")
    print(f"Task 2 diff: {TASK2_DIFFICULTY:.2f}")
    print(f"Output     : {OUTPUT_DIR}")
    print(f"MP4 record : {RECORD_VIDEO}")
    print("Screen recording mode: PyBullet GUI is paced like evaluate.py.")
    print()
    print("Task 1 model:")
    print(f"  {task1_model}")
    print("Task 1 norm:")
    print(f"  {task1_norm}")
    print("Task 2 model:")
    print(f"  {task2_model}")
    print("Task 2 norm:")
    print(f"  {task2_norm}")
    print()
    print("Camera:")
    print("  Initialized once per task.")
    print("  No camera following.")
    print("  No repeated resetDebugVisualizerCamera().")
    print("  Manual rotate / zoom / pan is preserved.")
    print("=" * 72)

    # --------------------------------------------------------
    # TASK 1
    # --------------------------------------------------------

    record_task(
        task_name="TASK 1 — PPO FLAT TERRAIN",
        model_path=task1_model,
        norm_path=task1_norm,
        env_factory=make_task1_env(),
        difficulty=None,
        output_path=task1_output,
    )

    # --------------------------------------------------------
    # TASK 2
    # --------------------------------------------------------

    record_task(
        task_name="TASK 2 — PPO UNEVEN TERRAIN",
        model_path=task2_model,
        norm_path=task2_norm,
        env_factory=make_task2_env(),
        difficulty=TASK2_DIFFICULTY,
        output_path=task2_output,
    )

    # --------------------------------------------------------
    # COMBINED
    # --------------------------------------------------------

    combine_videos(
        [task1_output, task2_output],
        combined_output,
    )

    print("\n" + "=" * 72)
    print("RECORDING COMPLETE")
    print("=" * 72)
    print(f"Task 1 : {task1_output}")
    print(f"Task 2 : {task2_output}")
    print(f"Combined: {combined_output}")
    print("=" * 72)


if __name__ == "__main__":
    main()

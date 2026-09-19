"""
Task 2 — terrain curriculum fine-tuning.

Starts from the validated Task-1 smooth PPO policy and gradually increases
terrain difficulty. Task-1 checkpoints are never overwritten.

Architecture:
PPO -> rate-limited q_des -> PD -> torque -> A1
"""

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.env_uneven_terrain import QuadrupedTerrainEnv


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 25_000
CHECKPOINT_FREQ = 50_000

# Conservative PPO fine-tuning settings.
# The Task-1 policy is already a strong locomotion policy, so Task-2
# training should make small updates instead of moving far from it.
LEARNING_RATE = 5e-5
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS = 5
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.10
ENT_COEF = 0.001
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.03

MODEL_DIR = PROJECT_ROOT / "checkpoints" / "task2_terrain"
BEST_DIR = PROJECT_ROOT / "models" / "task2_terrain"
LOG_DIR = PROJECT_ROOT / "logs" / "task2_terrain"

# Task 1 smooth policy used as the starting point for Task 2.
SOURCE_MODEL = (
    PROJECT_ROOT
    / "checkpoints"
    / "task1_flat"
    / "ppo_a1_task1_smooth_final.zip"
)

SOURCE_NORM = (
    PROJECT_ROOT
    / "checkpoints"
    / "task1_flat"
    / "ppo_a1_task1_smooth_final_vecnormalize.pkl"
)

for directory in (MODEL_DIR, BEST_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Terrain curriculum
# ---------------------------------------------------------------------------
# difficulty is deliberately conservative:
# transfer first, robustness second.
#
# The terrain difficulty is changed for the next environment reset.
STAGES = [
    (0, 0.05),
    (100_000, 0.10),
    (200_000, 0.15),
    (300_000, 0.20),
    (400_000, 0.25),
]


class TerrainCurriculumCallback(BaseCallback):
    """
    Gradually increases terrain difficulty during Task 2 training.

    Training and evaluation environments are kept at the same curriculum
    difficulty so that EvalCallback measures performance on the terrain
    currently being learned.
    """

    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.task2_start_timesteps = 0
        self.last_difficulty = None

    def _on_training_start(self):
        self.task2_start_timesteps = self.model.num_timesteps

    def _on_step(self) -> bool:
        elapsed = self.num_timesteps - self.task2_start_timesteps

        difficulty = STAGES[0][1]
        for start, value in STAGES:
            if elapsed >= start:
                difficulty = value

        # set_difficulty() changes the difficulty used when the next
        # terrain is generated. It does not regenerate terrain every step.
        if difficulty != self.last_difficulty:
            self.training_env.env_method("set_difficulty", difficulty)
            self.eval_env.env_method("set_difficulty", difficulty)
            self.last_difficulty = difficulty

            print(
                f"\n[TERRAIN CURRICULUM] "
                f"task2_steps={elapsed:,} "
                f"difficulty={difficulty:.2f}"
            )

        return True


def make_env():
    """
    Create a Task 2 uneven-terrain environment.

    The same factory is used for training and evaluation. The curriculum
    callback controls the actual difficulty.
    """

    def _make():
        env = QuadrupedTerrainEnv(
            difficulty=0.05,
            target_velocity=0.5,
            max_episode_steps=1000,
        )
        return Monitor(env)

    return _make


def main():
    print("=" * 72)
    print("TASK 2 — PPO + PD + TERRAIN CURRICULUM")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Verify Task 1 source artifacts before doing any expensive work.
    # -----------------------------------------------------------------------
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(
            f"Task 1 model not found:\n{SOURCE_MODEL}"
        )

    if not SOURCE_NORM.exists():
        raise FileNotFoundError(
            f"Task 1 VecNormalize file not found:\n{SOURCE_NORM}"
        )

    print("Starting from:", SOURCE_MODEL)
    print("VecNormalize:", SOURCE_NORM)
    print("Task 1 checkpoints are untouched.")
    print("Stages:", STAGES)
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print("Conservative PPO fine-tuning is enabled.")
    print()

    # -----------------------------------------------------------------------
    # Training environment
    # -----------------------------------------------------------------------
    train_raw = DummyVecEnv([make_env()])

    train_env = VecNormalize.load(
        str(SOURCE_NORM),
        train_raw,
    )

    train_env.training = True
    train_env.norm_reward = False
    train_env.clip_obs = 10.0

    # -----------------------------------------------------------------------
    # Evaluation environment
    # -----------------------------------------------------------------------
    eval_raw = DummyVecEnv([make_env()])

    eval_env = VecNormalize.load(
        str(SOURCE_NORM),
        eval_raw,
    )

    eval_env.training = False
    eval_env.norm_reward = False
    eval_env.clip_obs = 10.0

    # Evaluation must use the same observation normalization statistics
    # as the training environment.
    eval_env.obs_rms = train_env.obs_rms

    # -----------------------------------------------------------------------
    # Load the trained Task 1 PPO policy.
    # -----------------------------------------------------------------------
    model = PPO.load(
        str(SOURCE_MODEL),
        env=train_env,
        device="auto",
    )

    model.set_env(train_env)

    # -----------------------------------------------------------------------
    # Conservative PPO fine-tuning
    # -----------------------------------------------------------------------
    # PPO.load restores the Task-1 optimizer/hyperparameter state. We
    # deliberately override the update settings here so terrain adaptation
    # happens with small policy updates.
    model.n_steps = N_STEPS
    model.batch_size = BATCH_SIZE
    model.n_epochs = N_EPOCHS
    model.gamma = GAMMA
    model.gae_lambda = GAE_LAMBDA
    model.ent_coef = ENT_COEF
    model.vf_coef = VF_COEF
    model.max_grad_norm = MAX_GRAD_NORM
    model.target_kl = TARGET_KL

    # SB3 stores clip_range and learning-rate schedule internally.
    model.clip_range = lambda _: CLIP_RANGE
    model.lr_schedule = lambda _: LEARNING_RATE

    # PPO's optimizer was already created by PPO.load(), so update its
    # parameter-group learning rates explicitly.
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE

    print()
    print("Conservative PPO fine-tuning:")
    print(f"  learning_rate = {LEARNING_RATE}")
    print(f"  n_steps       = {N_STEPS}")
    print(f"  batch_size    = {BATCH_SIZE}")
    print(f"  n_epochs      = {N_EPOCHS}")
    print(f"  gamma         = {GAMMA}")
    print(f"  gae_lambda    = {GAE_LAMBDA}")
    print(f"  clip_range    = {CLIP_RANGE}")
    print(f"  ent_coef      = {ENT_COEF}")
    print(f"  target_kl     = {TARGET_KL}")
    print()

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    checkpoint = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_a1_task2_terrain",
        save_vecnormalize=True,
    )

    evaluator = EvalCallback(
        eval_env,
        best_model_save_path=str(BEST_DIR),
        log_path=str(BEST_DIR),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    curriculum = TerrainCurriculumCallback(eval_env)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[curriculum, checkpoint, evaluator],
        reset_num_timesteps=True,
        tb_log_name="PPO_A1_Task2_Terrain",
        progress_bar=True,
    )

    # -----------------------------------------------------------------------
    # Save final Task 2 model and normalization statistics.
    # -----------------------------------------------------------------------
    final_model = MODEL_DIR / "ppo_a1_task2_terrain_final"

    model.save(str(final_model))

    train_env.save(
        str(MODEL_DIR / "ppo_a1_task2_terrain_final_vecnormalize.pkl")
    )

    print("\n" + "=" * 72)
    print("TASK 2 TRAINING COMPLETE")
    print("=" * 72)
    print("Model:", final_model.with_suffix(".zip"))
    print(
        "Norm :",
        MODEL_DIR / "ppo_a1_task2_terrain_final_vecnormalize.pkl",
    )
    print("Best :", BEST_DIR)


if __name__ == "__main__":
    main()

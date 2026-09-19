"""
Task 1 — PPO training with smooth desired joint-position targets.

Architecture:
    PPO -> normalized action -> rate-limited desired joint position
        -> explicit PD -> torque -> PyBullet A1

This is a NEW experiment. It does not overwrite the old Task 1 model.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environments.env_flat_terrain import QuadrupedEnv


TOTAL_TIMESTEPS = 2_000_000
EVAL_FREQ = 50_000
CHECKPOINT_FREQ = 100_000

MODEL_DIR = PROJECT_ROOT / "checkpoints" / "task1_smooth"
BEST_DIR = PROJECT_ROOT / "models" / "task1_smooth"
LOG_DIR = PROJECT_ROOT / "logs" / "task1_smooth"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
BEST_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def make_train_env():
    def _make():
        env = QuadrupedEnv(
            terrain_id=0,
            render=False,
            difficulty=0.0,
            target_velocity=0.5,
            max_episode_steps=1000,
        )
        return Monitor(env)
    return _make


def make_eval_env():
    def _make():
        env = QuadrupedEnv(
            terrain_id=0,
            render=False,
            difficulty=0.0,
            target_velocity=0.5,
            max_episode_steps=1000,
        )
        return Monitor(env)
    return _make


def main():
    print("=" * 72)
    print("TASK 1 — PPO + PD + TARGET RATE LIMITER")
    print("=" * 72)
    print()
    print("NEW EXPERIMENT: task1_smooth")
    print("Old checkpoints are NOT overwritten.")
    print()
    print("PPO -> rate-limited q_des -> PD -> torque -> A1")
    print()

    train_env_raw = DummyVecEnv([make_train_env()])
    train_env = VecNormalize(
        train_env_raw,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )

    eval_env_raw = DummyVecEnv([make_eval_env()])
    eval_env = VecNormalize(
        eval_env_raw,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )

    eval_env.obs_rms = train_env.obs_rms

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(MODEL_DIR),
        name_prefix="ppo_a1_task1_smooth",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(BEST_DIR),
        log_path=str(BEST_DIR),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 256],
                vf=[256, 256],
            )
        ),
        tensorboard_log=str(LOG_DIR),
        verbose=1,
        device="auto",
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )

        final_model = MODEL_DIR / "ppo_a1_task1_smooth_final"
        final_norm = MODEL_DIR / "ppo_a1_task1_smooth_final_vecnormalize.pkl"

        model.save(str(final_model))
        train_env.save(str(final_norm))

        print()
        print("=" * 72)
        print("TRAINING COMPLETE")
        print("=" * 72)
        print(f"Model : {final_model}.zip")
        print(f"Norm  : {final_norm}")
        print(f"Best  : {BEST_DIR}")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()

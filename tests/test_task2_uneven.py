import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from stable_baselines3.common.env_checker import check_env
from environments.env_uneven_terrain import QuadrupedTerrainEnv

print("="*68)
print("TASK 2 ENVIRONMENT VALIDATION")
print("="*68)

env = QuadrupedTerrainEnv(difficulty=0.15, target_velocity=0.5, max_episode_steps=1000)

print("[1] Gymnasium/SB3 check_env")
check_env(env, warn=True)
print("PASS")

print("\n[2] Reset + finite observation")
obs, info = env.reset(seed=0)
assert obs.shape == (55,)
assert np.all(np.isfinite(obs))
print("PASS")

print("\n[3] Zero-action PD hold on low terrain")
max_tau=0; min_z=999; max_roll=0; max_pitch=0
for _ in range(300):
    obs, r, term, trunc, info = env.step(np.zeros(12,dtype=np.float32))
    max_tau=max(max_tau,info["max_torque"])
    min_z=min(min_z,info["base_height"])
    max_roll=max(max_roll,abs(info["roll"]))
    max_pitch=max(max_pitch,abs(info["pitch"]))
    assert not term, "Robot fell during low-difficulty PD hold"
print(f"min_z={min_z:.3f} max_roll={max_roll:.3f} max_pitch={max_pitch:.3f} max_tau={max_tau:.2f}")
print("PASS")

print("\n[4] Terrain generation at several difficulties")
for d in (0.25,0.50,0.75,1.00):
    env.set_difficulty(d)
    obs, info = env.reset(seed=123)
    assert obs.shape == (55,)
    assert np.all(np.isfinite(obs))
    print(f"difficulty={d:.2f} reset_z={info['base_height']:.3f}")
print("PASS")

env.close()
print("\nSTATUS: TASK 2 ENVIRONMENT READY FOR TRAINING")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.climb_env import ClimbEnv


# 環境の作成（ベクトル化）
env = DummyVecEnv([lambda: ClimbEnv(render=False)])

# モデルの作成（PPO + MLPポリシー）
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2
)

# 学習実行
model.learn(total_timesteps=100_000)

# モデルの保存
model.save("stickman_climb_model")


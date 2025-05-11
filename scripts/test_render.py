from stable_baselines3 import PPO

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.climb_env import ClimbEnv

import time

# 1. 環境をGUIモードで起動
env = ClimbEnv(render=True)

# 2. 学習済みモデルを読み込み
model = PPO.load("stickman_climb_model")

# 3. 環境をリセットして初期状態を取得
obs = env.reset()

# 4. モデルに従って行動しながら可視化
for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    time.sleep(1/60)  # 表示速度調整（1秒間に60フレーム）

    if done:
        print("エピソード終了（コイン接触 or 落下）")
        obs = env.reset()

# 5. 終了処理
env.close()


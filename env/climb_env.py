import gym
import pybullet as p
import pybullet_data
import numpy as np

class ClimbEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()

        # PyBulletサーバーの起動
        if render:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # 地面
        p.loadURDF("plane.urdf")

        # 三段構造を生成
        self._create_platforms_and_coin()

        # 棒人間をロード
        self.robot_id = p.loadURDF("stickman/stickman.urdf", basePosition=[-2.0, 0, 1.0])

        # 動かせる関節番号を取得
        self.joint_indices = [
            i for i in range(p.getNumJoints(self.robot_id))
            if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
        ]

        # アクション・観測空間
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(self.joint_indices),), dtype=np.float32)
        obs_dim = len(self.joint_indices) * 2 + 6
        high = np.ones(obs_dim) * np.inf
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # 状態変数
        self.fall_count = 0
        self.reached_tiers = [False, False, False]

    def _create_platforms_and_coin(self):
        def create_tier(radius, height, z_offset, color):
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
            return p.createMultiBody(0, col, vis, [0, 0, z_offset])

        self.tier1_height = 0.2
        self.tier2_height = 0.2
        self.tier3_height = 0.2

        create_tier(1.0, self.tier1_height, self.tier1_height/2, [0.2, 0.4, 0.8, 1])
        create_tier(0.6, self.tier2_height, self.tier1_height + self.tier2_height/2, [0.2, 0.6, 0.6, 1])
        create_tier(0.3, self.tier3_height, self.tier1_height + self.tier2_height + self.tier3_height/2, [0.4, 0.8, 0.4, 1])

        self.coin_radius = 0.05
        coin_z = self.tier1_height + self.tier2_height + self.tier3_height + self.coin_radius
        coin_col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.coin_radius)
        coin_vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.coin_radius, rgbaColor=[1, 1, 0, 1])
        self.coin_id = p.createMultiBody(0, coin_col, coin_vis, [0, 0, coin_z])

    def reset(self):
        self.fall_count = 0
        self.reached_tiers = [False, False, False]

        p.resetBasePositionAndOrientation(self.robot_id, [-2.0, 0, 1.0], [0, 0, 0, 1])
        for j in self.joint_indices:
            name = p.getJointInfo(self.robot_id, j)[1].decode("utf-8")
            if "hip" in name:
                p.resetJointState(self.robot_id, j, targetValue=0.2, targetVelocity=0.0)
            elif "knee" in name:
                p.resetJointState(self.robot_id, j, targetValue=-0.4, targetVelocity=0.0)
            else:
                p.resetJointState(self.robot_id, j, targetValue=0.0, targetVelocity=0.0)

        return self._get_observation()

    def _get_observation(self):
        joint_states = p.getJointStates(self.robot_id, self.joint_indices)
        joint_pos = [s[0] for s in joint_states]
        joint_vel = [s[1] for s in joint_states]
        torso_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        coin_pos, _ = p.getBasePositionAndOrientation(self.coin_id)
        rel_pos = [coin_pos[i] - torso_pos[i] for i in range(3)]
        return np.array(joint_pos + joint_vel + list(torso_pos) + rel_pos, dtype=np.float32)

    def step(self, action):
        max_speed = 5.0
        max_force = 50.0
        for i, j in enumerate(self.joint_indices):
            target_vel = float(np.clip(action[i], -1, 1)) * max_speed
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, targetVelocity=target_vel, force=max_force)

        p.stepSimulation()
        obs = self._get_observation()
        reward, done = self._compute_reward()
        return obs, reward, done, {}

    def _compute_reward(self):
        reward = 0.0
        done = False

        torso_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        coin_pos, _ = p.getBasePositionAndOrientation(self.coin_id)

        # 成功報酬（コインに触れる）
        contacts = p.getContactPoints(self.robot_id, self.coin_id)
        if len(contacts) > 0:
            reward += 100.0
            done = True
            return reward, done

        # 段差到達ごとの報酬
        height = torso_pos[2]
        thresholds = [
            (self.tier1_height, 10),
            (self.tier1_height + self.tier2_height, 20),
            (self.tier1_height + self.tier2_height + self.tier3_height, 30)
        ]
        for i, (th, rew) in enumerate(thresholds):
            if height > th and not self.reached_tiers[i]:
                reward += rew
                self.reached_tiers[i] = True

        # 距離 shaping
        dist = np.linalg.norm(np.array(torso_pos) - np.array(coin_pos))
        reward += -0.1 * dist

        # 傾き shaping（姿勢ペナルティ）
        orientation = p.getBasePositionAndOrientation(self.robot_id)[1]
        angle_penalty = abs(orientation[0]) + abs(orientation[1])
        reward += -2.0 * angle_penalty

        # 転倒判定
        if height < 0.2:
            self.fall_count += 1
            reward -= 50.0
            if self.fall_count >= 3:
                done = True

        return reward, done


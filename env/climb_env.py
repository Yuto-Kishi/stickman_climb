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
        self.robot_id = p.loadURDF("stickman/stickman.urdf", basePosition=[-1.0, 0, 1.0])

        # 動かせる関節番号を取得（固定関節以外）
        self.joint_indices = [
            i for i in range(p.getNumJoints(self.robot_id))
            if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED
        ]

        # アクション空間（各関節に -1.0〜1.0 の速度指令）
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(self.joint_indices),), dtype=np.float32)

        # 観測空間（仮のサイズ：角度・速度・ロボット座標・コイン座標）
        obs_dim = len(self.joint_indices) * 2 + 6
        high = np.ones(obs_dim) * np.inf
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def _create_platforms_and_coin(self):
        """段差とコインの生成"""
        def create_tier(radius, height, z_offset, color):
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
            return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                     basePosition=[0, 0, z_offset])

        self.tier1_height = 0.2
        self.tier2_height = 0.2
        self.tier3_height = 0.2

        create_tier(1.0, self.tier1_height, self.tier1_height/2, [0.2, 0.4, 0.8, 1])
        create_tier(0.6, self.tier2_height, self.tier1_height + self.tier2_height/2, [0.2, 0.6, 0.6, 1])
        create_tier(0.3, self.tier3_height, self.tier1_height + self.tier2_height + self.tier3_height/2, [0.4, 0.8, 0.4, 1])

        # コイン（球体）
        self.coin_radius = 0.05
        coin_z = self.tier1_height + self.tier2_height + self.tier3_height + self.coin_radius
        coin_col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.coin_radius)
        coin_vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.coin_radius, rgbaColor=[1, 1, 0, 1])
        self.coin_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=coin_col,
                                         baseVisualShapeIndex=coin_vis,
                                         basePosition=[0, 0, coin_z])

    def reset(self):
        # 棒人間の初期位置をリセット
        p.resetBasePositionAndOrientation(self.robot_id, [-1.0, 0, 1.0], [0, 0, 0, 1])
        for j in self.joint_indices:
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
        # アクションを関節に適用（velocity control）
        max_speed = 5.0
        max_force = 50.0
        for i, j in enumerate(self.joint_indices):
            target_vel = float(np.clip(action[i], -1, 1)) * max_speed
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_vel,
                force=max_force
            )

        # シミュレーション1ステップ実行
        p.stepSimulation()

        # 観測を取得
        obs = self._get_observation()

        # 報酬と終了判定
        reward, done = self._compute_reward()

        return obs, reward, done, {}

    def _compute_reward(self):
        reward = 0.0
        done = False

        # トルソーとコインの位置取得
        torso_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        coin_pos, _ = p.getBasePositionAndOrientation(self.coin_id)

        # コインに触れたか確認
        contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.coin_id)
        if len(contacts) > 0:
            reward += 100.0
            done = True
        else:
            # shaping: 高さ or コインへの接近
            dist = np.linalg.norm(np.array(torso_pos) - np.array(coin_pos))
            reward += -0.1 * dist

        # 落下判定（地面に落ちたら終了）
        if torso_pos[2] < 0.2:
            reward -= 50.0
            done = True

        return reward, done


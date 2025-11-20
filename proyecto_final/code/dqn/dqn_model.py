# dqn_model.py

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.monitor import Monitor

def build_dqn(env, n_stack=16):
    env = Monitor(env, filename="episode_log") 
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=n_stack)

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=0.02,
        buffer_size=100_000,
        learning_starts=100_000,
        batch_size=32,
        tau=1.0,
        gamma=0.993,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.01,
        verbose=1,
    )

    return model

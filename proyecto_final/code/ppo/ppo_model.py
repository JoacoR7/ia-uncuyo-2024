# ppo_model.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from atari_env import create_space_invaders_env
import numpy as np
import imageio
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import ClipRewardEnv


def build_ppo(env, n_stack=4, tensorboard_log="./ppo_tensorboard/"):
    env = ClipRewardEnv(env)               # Clip de recompensas [-1,1]
    env = Monitor(env, filename="episode_log")
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=n_stack)

    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        learning_rate=1e-5,   # más estable
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.01,        # exploración
        device="cuda"
    )
    return model

def evaluate_model(model_path, n_eval_episodes=10, render=False, n_stack=4):
    """
    Evalúa un modelo PPO entrenado.
    
    Args:
        model_path: Ruta al modelo guardado (.zip)
        n_eval_episodes: Número de episodios para evaluar
        render: Si True, renderiza el ambiente
        n_stack: Número de frames a apilar
    
    Returns:
        dict con estadísticas de evaluación
    """
    # Crear ambiente de evaluación
    render_mode = "human" if render else "rgb_array"
    env = create_space_invaders_env(render_mode=render_mode)
    
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=n_stack)
    
    # Cargar modelo
    model = PPO.load(model_path, env=env)
    
    # Evaluar
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    
    print(f"\n{'='*50}")
    print(f"Evaluación del modelo: {model_path}")
    print(f"{'='*50}")
    print(f"Episodios evaluados: {n_eval_episodes}")
    print(f"Recompensa promedio: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"{'='*50}\n")
    
    env.close()
    
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "n_episodes": n_eval_episodes
    }

def evaluate_detailed(model_path, n_eval_episodes=5, n_stack=4):
    """
    Evaluación detallada con estadísticas por episodio.
    
    Returns:
        dict con listas de recompensas, longitudes y otras métricas
    """
    env = create_space_invaders_env(render_mode="rgb_array")
    
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=n_stack)
    
    model = PPO.load(model_path, env=env)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episodio {episode+1}/{n_eval_episodes} - "
              f"Recompensa: {episode_reward:.2f}, "
              f"Longitud: {episode_length}")
    
    env.close()
    
    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "mean_length": np.mean(episode_lengths)
    }
    
    return results

def record_gameplay(model_path, output_filename="ppo_gameplay.mp4", n_stack=4):
    """
    Graba un video del agente jugando.
    
    Args:
        model_path: Ruta al modelo guardado
        output_filename: Nombre del archivo de video
        n_stack: Número de frames a apilar
    """
    env = create_space_invaders_env(render_mode="rgb_array")
    
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=n_stack)
    
    model = PPO.load(model_path, env=env)
    
    obs = env.reset()
    frames = []
    done = False
    total_reward = 0
    
    while not done:
        # Renderizar frame original del ambiente (antes del frame stacking)
        frame = env.envs[0].render()
        frames.append(frame)
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
    
    env.close()
    
    # Guardar video
    imageio.mimsave(output_filename, frames, fps=30, format="FFMPEG")
    print(f"Video guardado en {output_filename}")
    print(f"Recompensa total del episodio: {total_reward:.2f}")
    
    return total_reward

def plot_evaluation_results(results, save_path="evaluation_plot.png"):
    """
    Genera gráficos de los resultados de evaluación.
    
    Args:
        results: dict retornado por evaluate_detailed()
        save_path: Ruta para guardar el gráfico
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico de recompensas por episodio
    axes[0].plot(results["episode_rewards"], marker='o', linewidth=2)
    axes[0].axhline(results["mean_reward"], color='r', linestyle='--', 
                    label=f'Media: {results["mean_reward"]:.2f}')
    axes[0].set_xlabel("Episodio")
    axes[0].set_ylabel("Recompensa")
    axes[0].set_title("Recompensas por Episodio")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico de longitudes por episodio
    axes[1].plot(results["episode_lengths"], marker='s', linewidth=2, color='green')
    axes[1].axhline(results["mean_length"], color='r', linestyle='--',
                    label=f'Media: {results["mean_length"]:.0f}')
    axes[1].set_xlabel("Episodio")
    axes[1].set_ylabel("Longitud (pasos)")
    axes[1].set_title("Longitud de Episodios")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Gráfico guardado en {save_path}")
    plt.close()
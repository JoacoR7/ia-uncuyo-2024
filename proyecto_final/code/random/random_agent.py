from atari_env import (
    create_space_invaders_env,
    create_clipped_space_invaders_env
)
import csv
import time
import numpy as np

def evaluate_random_agent_to_csv(
    csv_path,
    num_episodes=10,
    use_clipped=False,
):
    # Crear entorno según corresponda
    if use_clipped:
        env = create_clipped_space_invaders_env()
    else:
        env = create_space_invaders_env()

    # Crear archivo CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_number", "episode_reward", "episode_length", "episode_time_seconds"])

    rewards = []
    lengths = []

    print(f"Evaluando agente random por {num_episodes} episodios...")

    for ep in range(num_episodes):
        start_time = time.time()

        obs, info = env.reset()
        done = False
        total_reward = 0
        length = 0

        while not done:
            action = env.action_space.sample()  # acción aleatoria
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            length += 1
            done = terminated or truncated

        duration = time.time() - start_time

        # Guardar en listas
        rewards.append(total_reward)
        lengths.append(length)

        # Guardar en CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, total_reward, length, duration])

        print(f"[Random Eval {ep+1}/{num_episodes}] Reward={total_reward}, Length={length}, Time={duration:.2f}s")

    print("\n--- RESULTADOS RANDOM ---")
    print("Recompensas:", rewards)
    print("Longitudes:", lengths)
    print("Promedio recompensa:", np.mean(rewards))
    print("Promedio longitud:", np.mean(lengths))

    return rewards, lengths
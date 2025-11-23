import numpy as np
import time
import csv

from atari_env import create_space_invaders_env
from agent.discretizer import aggregate_and_discretize_strip, convert_state_vector_to_int
from agent.epsilon_policy import choose_action_epsilon_greedy
from agent.q_table import create_q_table

def train_q_learning(
    num_episodes,
    num_state_variables,
    num_levels,
    save_path,
    alpha,
    gamma,
    epsilon,
    min_epsilon,
    decay
):

    env = create_space_invaders_env()
    num_states = num_levels ** num_state_variables
    num_actions = env.action_space.n

    q_table = create_q_table(num_states, num_actions)

    rewards_per_episode = []

    csv_filename = save_path + "q_learning_metrics.csv"
    with open(csv_filename, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "reward", "length", "time_sec"])

    for episode in range(num_episodes):

        start = time.time()
        obs, info = env.reset()
        done = False
        total_reward = 0
        length = 0

        strips = env.crop_strips(obs, num_vertical=8)
        disc_vector = [aggregate_and_discretize_strip(s, num_levels) for s in strips]
        state = convert_state_vector_to_int(disc_vector, num_levels)

        while not done:

            length += 1
            action = choose_action_epsilon_greedy(q_table, state, epsilon, num_actions)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            next_strips = env.crop_strips(next_obs, num_vertical=8)
            next_disc_vector = [aggregate_and_discretize_strip(s, num_levels) for s in next_strips]
            next_state = convert_state_vector_to_int(next_disc_vector, num_levels)

            q_table[state, action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            state = next_state
            done = terminated or truncated

        epsilon = max(min_epsilon, epsilon * decay)
        rewards_per_episode.append(total_reward)

        with open(csv_filename, "a", newline="") as f:
            csv.writer(f).writerow([episode, total_reward, length, time.time() - start])

        if (episode + 1) % 1000 == 0:
            np.save(save_path + f"q_table_{episode+1}.npy", q_table)
            print(f"[{episode}] Guardada Q-table")

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} - Reward={total_reward:.1f} - Epsilon={epsilon:.4f}")

    return q_table, rewards_per_episode

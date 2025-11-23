import numpy as np

def choose_action_epsilon_greedy(q_table, state_idx, epsilon, num_actions):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(num_actions)
    return np.argmax(q_table[state_idx])

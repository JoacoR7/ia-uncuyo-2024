from training.train_q_learning import train_q_learning

if __name__ == "__main__":
    train_q_learning(
        num_episodes=20000,
        num_state_variables= 8,
        num_levels= 4,
        save_path="./results/",
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        min_epsilon = 0.2,
        decay = 0.999
    )

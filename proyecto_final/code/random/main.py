from random_agent import *

if __name__ == "__main__":
    evaluate_random_agent_to_csv(
        "/content/random_results.csv",
        num_episodes=100,
        use_clipped=True
    )

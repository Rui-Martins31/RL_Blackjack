import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from easy21 import Easy21
import _config
from visualize import plot_policy_heatmap, plot_value_function_3d

def main(training: bool = False):
    # Environment
    env: Easy21  = Easy21()
    observation  = env.reset()

    # Action
    agent: Agent = Agent(
        observation=observation,
        num_actions=2
    )
    if training: agent.train()
    else: agent.test(); agent.load_q_matrix()

    # Run multiple episodes
    num_episodes: int    = _config.NUM_EPISODES
    total_rewards: float = 0.0
    list_rewards: list[float] = []
    list_wins: int       = 0

    for episode in range(num_episodes):
        # Reset the environment
        observation           = env.reset()
        agent.reset(observation)

        episode_reward: float = 0.0
        terminated: bool      = False

        print(f"\n--- Episode {episode + 1} ---")
        print(f"Initial observation: {observation}")
        print(f"  Player sum: {observation[0]}")
        print(f"  Dealer showing: {observation[1]}")

        while not terminated:
            # Select action
            action = agent.select()

            if action: action_name: str = "HIT"
            else: action_name: str = "STICK"
            print(f"Action: {action_name}")

            # Take action in the environment
            observation, terminated, reward = env.step(action)

            if not terminated: print(f"New player sum: {observation[0]}")

            # Update Q-matrix
            if not agent.update(observation, int(action), float(reward), terminated):
                terminated = True

            # Store reward
            episode_reward = reward

        # Print episode results
        print(f"\nFinal Observation: {observation}")
        print(f"Final reward: {episode_reward}")
        if episode_reward > 0:
            print("Result: WIN")
            list_wins += 1
        elif episode_reward < 0:
            print("Result: LOSS")
        else:
            print("Result: DRAW")

        total_rewards += episode_reward
        list_rewards.append(total_rewards)

    # Print summary statistics
    print(f"\n{'='*40}")
    print(f"Summary after {num_episodes} episodes:")
    print(f"Total rewards: {total_rewards}")
    print(f"Average reward: {total_rewards/num_episodes:.2f}")
    print(f"Wins (%): {list_wins/num_episodes * 100:.2f}%")

    print(f"Q-Matrix: {np.amax(agent.q_matrix)}")
    print(f"{'='*40}")

    # Save Q-matrix
    if training: agent.save_q_matrix(f"{_config.PATH_SAVE_Q_MATRIX}q_matrix.pkl")

    # Plot policy heatmap
    print("\nGenerating policy heatmap...")
    plot_policy_heatmap(agent.q_matrix, save_path=f"{_config.PATH_SAVE_IMAGES}policy_heatmap.png")
    plt.show()

    # Plot 3D value function
    print("\nGenerating 3D value function plot...")
    plot_value_function_3d(agent.q_matrix, save_path=f"{_config.PATH_SAVE_IMAGES}value_function_3d.png")
    plt.show()


if __name__ == "__main__":
    main(training=_config.AGENT_TRAIN)
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from easy21 import Easy21
import _config

def main():
    # Environment
    env: Easy21  = Easy21()
    observation  = env.reset()

    # Action
    agent: Agent = Agent(
        observation=observation,
        num_actions=2
    )
    agent.train()

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

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(list_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards per Episode')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save model
    


if __name__ == "__main__":
    main()
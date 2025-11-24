import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
import _config

# Environment
env: gym.Env       = gym.make('Blackjack-v1')#, render_mode='human')
observation, info  = env.reset()

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
list_wins: int = 0

for episode in range(num_episodes):
    # Reset the environment
    observation, info     = env.reset()
    agent.reset(observation)

    episode_reward: float = 0.0
    done: bool            = False

    print(f"\n--- Episode {episode + 1} ---")
    print(f"Initial observation: {observation}")
    print(f"  Player sum: {observation[0]}")
    print(f"  Dealer showing: {observation[1]}")
    print(f"  Usable ace: {observation[2]}")

    while not done:
        # Select action
        action = agent.select()

        if action: action_name: str = "HIT"
        else: action_name: str = "STICK"
        print(f"Action: {action_name}")

        # Take action in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if not done: print(f"New player sum: {observation[0]}")

        # Update Q-matrix
        if not agent.update(observation, int(action), float(reward), done):
            done = True

        # Store reward
        episode_reward = reward

    # Print episode results
    print(f"\nFinal reward: {episode_reward}")
    if episode_reward > 0:
        print("Result: WIN")
    elif episode_reward < 0:
        print("Result: LOSS")
    else:
        print("Result: DRAW")

    # Track statistics
    if episode_reward > 0:
        list_wins += 1

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

# Close the environment
env.close()
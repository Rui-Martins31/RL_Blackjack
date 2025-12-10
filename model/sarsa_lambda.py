import numpy as np
import matplotlib.pyplot as plt
import pickle

from agent import Agent
from easy21 import Easy21
import _config


def load_optimal_q_matrix(filepath: str = None):
    if filepath is None:
        filepath   = "q_matrix/monte_carlo_q_matrix.pkl"

    try:
        with open(filepath, 'rb') as f:
            q_star = pickle.load(f)
        print(f"Loaded optimal Q-matrix from {filepath}")
        return q_star
    except FileNotFoundError:
        print(f"ERROR: Optimal Q-matrix not found at {filepath}")
        print("Please run Monte Carlo simulation first to generate Q*.")
        return None


def calculate_mse(q_matrix, q_star):
    return np.mean((q_matrix - q_star) ** 2)


def run_sarsa_lambda_episode(env, agent):
    observation = env.reset()
    agent.reset(observation)
    terminated  = False

    while not terminated:
        action  = agent.select()
        observation, terminated, reward = env.step(action)
        agent.update(observation, int(action), float(reward), terminated)


def run_sarsa_lambda_experiment(lambda_param, q_star, num_episodes=1000, track_learning_curve=False):
    env         = Easy21()
    observation = env.reset()

    agent = Agent(
        observation=observation,
        num_actions=2,
        monte_carlo=False,
        lambda_param=lambda_param
    )
    agent.train()

    learning_curve = [] if track_learning_curve else None

    for episode in range(num_episodes):
        run_sarsa_lambda_episode(env, agent)

        if track_learning_curve:
            mse = calculate_mse(agent.q_matrix, q_star)
            learning_curve.append(mse)

    final_mse = calculate_mse(agent.q_matrix, q_star)
    return final_mse, learning_curve


def plot_mse_vs_lambda(lambda_values, mse_values, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, mse_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Lambda (λ)', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title('MSE vs Lambda for Sarsa(λ)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"MSE vs Lambda plot saved to {save_path}")


def plot_learning_curves(lambda_0_curve, lambda_1_curve, save_path=None):
    plt.figure(figsize=(12, 6))
    episodes = range(1, len(lambda_0_curve) + 1)

    plt.plot(episodes, lambda_0_curve, 'b-', linewidth=1.5, label='λ = 0 (Sarsa)', alpha=0.8)
    plt.plot(episodes, lambda_1_curve, 'r-', linewidth=1.5, label='λ = 1 (Monte Carlo)', alpha=0.8)

    plt.xlabel('Episode Number', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title('Learning Curves: MSE vs Episode Number', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves plot saved to {save_path}")


def main():

    q_star = load_optimal_q_matrix()
    if q_star is None:
        return

    lambda_values = [i / 10.0 for i in range(11)]
    print(f"\nTesting lambda values: {lambda_values}")

    mse_values: list = []
    lambda_0_curve   = None
    lambda_1_curve   = None

    print("\nRunning Sarsa(λ)...")
    for idx, lambda_param in enumerate(lambda_values):
        print(f"[{idx+1}/{len(lambda_values)}] Testing λ = {lambda_param:.1f}...")
        track_curve = (lambda_param == 0.0 or lambda_param == 1.0)

        final_mse, learning_curve = run_sarsa_lambda_experiment(
            lambda_param=lambda_param,
            q_star=q_star,
            num_episodes=1000,
            track_learning_curve=track_curve
        )

        mse_values.append(final_mse)

        if lambda_param   == 0.0:
            lambda_0_curve = learning_curve
        elif lambda_param == 1.0:
            lambda_1_curve = learning_curve

        print(f"  λ = {lambda_param:.1f}: MSE = {final_mse:.6f}")

    print("\n" + "="*60)
    print("Results Summary:")
    print("="*60)
    for lambda_param, mse in zip(lambda_values, mse_values):
        print(f"  λ = {lambda_param:.1f}: MSE = {mse:.6f}")

    best_idx    = np.argmin(mse_values)
    best_lambda = lambda_values[best_idx]
    best_mse    = mse_values[best_idx]
    print(f"\n  Best λ = {best_lambda:.1f} with MSE = {best_mse:.6f}")
    print("="*60)

    print("\nGenerating plots...")

    plot_mse_vs_lambda(
        lambda_values,
        mse_values,
        save_path="images/td_learning_mse_vs_lambda.png"
    )

    if lambda_0_curve and lambda_1_curve:
        plot_learning_curves(
            lambda_0_curve,
            lambda_1_curve,
            save_path="images/td_learning_curves.png"
        )

    print("\n")
    print("="*60)

    plt.show()


if __name__ == "__main__":
    main()

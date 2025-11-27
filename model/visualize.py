import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_policy_heatmap(q_matrix: np.ndarray, save_path: str = None):
    """
    Plot a heatmap showing the optimal policy (best action for each state).
    Both axes start from the bottom left corner.

    Args:
        q_matrix: Q-matrix with shape (num_actions, dealer_card, player_sum)
                  Note: Matrix is indexed from 0, but index 0 is unused.
                  Actual values start at index 1 (dealer=1, player=1)
        save_path: Optional path to save the figure
    """
    # Get the best action for each state
    # argmax over action dimension (axis=0)
    policy = np.argmax(q_matrix, axis=0)

    # Skip index 0 (unused) and use indices 1 onwards
    policy_trimmed = policy[1:, 1:]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap with origin at bottom left
    # Axes represent actual card/sum values (starting from 1)
    im = ax.imshow(
        policy_trimmed.T,
        cmap='coolwarm',
        aspect='auto',
        origin='lower',  # Start from bottom left
        extent=[0.5, policy_trimmed.shape[0] + 0.5, 0.5, policy_trimmed.shape[1] + 0.5],
        vmin=0,
        vmax=1
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_label('Action (0=STICK, 1=HIT)', fontsize=12)

    # Set tick positions to show actual values (1-10 for dealer, 1-10 for player)
    ax.set_xticks(range(1, policy_trimmed.shape[0] + 1))
    ax.set_yticks(range(1, policy_trimmed.shape[1] + 1))

    # Labels and title
    ax.set_xlabel('Dealer Showing', fontsize=12)
    ax.set_ylabel('Player Sum', fontsize=12)
    ax.set_title('Optimal Policy', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy heatmap saved to {save_path}")

    return fig


def plot_value_function_3d(q_matrix: np.ndarray, save_path: str = None):
    """
    Plot a 3D surface of the value function V*(s) = max_a Q*(s,a).
    Similar to the Sutton & Barto Blackjack example.

    Args:
        q_matrix: Q-matrix with shape (num_actions, dealer_card, player_sum)
                  Note: Matrix is indexed from 0, but index 0 is unused.
                  Actual values start at index 1 (dealer=1, player=1)
        save_path: Optional path to save the figure
    """
    # Get max Q-value for each state (value function)
    value_function = np.max(q_matrix, axis=0)

    # Skip index 0 (unused) and use indices 1 onwards
    value_function_trimmed = value_function[1:, 1:]

    # Create meshgrid for 3D plot with actual card/sum values
    dealer_range = np.arange(1, value_function_trimmed.shape[0] + 1)
    player_range = np.arange(1, value_function_trimmed.shape[1] + 1)
    X, Y = np.meshgrid(dealer_range, player_range)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(
        X, Y, value_function_trimmed.T,
        cmap='viridis',
        edgecolor='none',
        alpha=0.8,
        antialiased=True
    )

    # Labels and title
    ax.set_xlabel('Dealer Showing', fontsize=11, labelpad=10)
    ax.set_ylabel('Player Sum', fontsize=11, labelpad=10)
    ax.set_zlabel('Value', fontsize=11, labelpad=10)
    ax.set_title('Optimal Value Function V*(s)', fontsize=14, fontweight='bold', pad=20)

    # Set view angle (similar to Sutton & Barto)
    ax.view_init(elev=30, azim=45)

    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D value function plot saved to {save_path}")

    return fig

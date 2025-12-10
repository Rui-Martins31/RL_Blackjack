import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_policy_heatmap(q_matrix: np.ndarray, save_path: str = None):
    # Get best action
    policy = np.argmax(q_matrix, axis=0)

    # Offset
    policy_trimmed = policy[1:, 1:]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Heatmap
    im = ax.imshow(
        policy_trimmed.T,
        cmap='coolwarm',
        aspect='auto',
        origin='lower',
        extent=[1, policy_trimmed.shape[0] + 1, 1, policy_trimmed.shape[1] + 1],
        vmin=0,
        vmax=1
    )

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_label('Action (0=STICK, 1=HIT)', fontsize=12)

    ax.set_xticks(range(1, policy_trimmed.shape[0] + 1))
    ax.set_yticks(range(1, policy_trimmed.shape[1] + 1))

    ax.set_xlabel('Dealer Showing', fontsize=12)
    ax.set_ylabel('Player Sum', fontsize=12)
    ax.set_title('Optimal Policy', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy heatmap saved to {save_path}")

    return fig


def plot_value_function_3d(q_matrix: np.ndarray, save_path: str = None, mask_unvisited: bool = True):
    # Get max Q-value
    value_function = np.max(q_matrix, axis=0)

    # Offset
    value_function_trimmed = value_function[1:, 1:]

    if mask_unvisited:
        visited_mask           = np.any(q_matrix[:, 1:, 1:] != 0, axis=0)
        value_function_trimmed = np.where(visited_mask, value_function_trimmed, np.nan)

    # Meshgrid
    player_range = np.arange(1, value_function_trimmed.shape[1] + 1)
    dealer_range = np.arange(1, value_function_trimmed.shape[0] + 1)

    Y, X = np.meshgrid(dealer_range, player_range)

    # 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection='3d')

    # Plot surface
    surf = ax.plot_surface(
        X, Y, value_function_trimmed.T,
        cmap='viridis',
        edgecolor='none',
        alpha=0.8,
        antialiased=True
    )

    # Labels and title
    ax.set_xlabel('Player Sum', fontsize=11, labelpad=10)
    ax.set_ylabel('Dealer Showing', fontsize=11, labelpad=10)
    ax.set_zlabel('Value', fontsize=11, labelpad=10)
    title = 'Optimal Value Function V*(s)'
    if mask_unvisited:
        title += ' (unvisited states masked)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    ax.set_xlim(1, player_range[-1])
    ax.set_ylim(1, dealer_range[-1])

    ax.set_yticks(dealer_range)
    ax.set_xticks(player_range[::2])

    ax.invert_yaxis()

    ax.view_init(elev=30, azim=225)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D value function plot saved to {save_path}")

    return fig
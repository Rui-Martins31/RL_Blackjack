# Scenario
SCENARIO_NAME: str                    = "Blackjack-v1"
SCENARIO_OBSERVATIONS: tuple[int]     = (0, 0, 0)
SCENARIO_OBSERVATIONS_NUM: tuple[int] = (31+1, 31+1, 2)         # +1 is the offset since array starts at [0]
SCENARIO_ACTIONS: tuple[int]          = (0, 1)
SCENARIO_ACTIONS_NUM: int             = len(SCENARIO_ACTIONS)

# Agent
AGENT_TRAIN: bool = True

# Episode
NUM_EPISODES: int = 1_000_000

# Training
EPSILON: float              = 1.0
EPSILON_DECAY_FACTOR: float = 0.99
EPSILON_MIN: float          = 0.1
ALPHA: float                = 0.1    # Learning Rate
GAMMA: float                = 0.99   # Discount Factor

"""
Scenario: Blackjack-v1

Observation space: [my_card, dealer's_card, usable_ace]
Action space: [0 (Stick), 1 (Hit)]
"""
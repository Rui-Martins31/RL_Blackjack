# Scenario
SCENARIO_OBSERVATIONS: tuple[int]     = (0, 0)
SCENARIO_OBSERVATIONS_NUM: tuple[int] = (21+1, 21+1)         # +1 is the offset since array starts at [0]
SCENARIO_ACTIONS: tuple[int]          = (0, 1)
SCENARIO_ACTIONS_NUM: int             = len(SCENARIO_ACTIONS)

# Agent
AGENT_TRAIN: bool = True

# Episode
NUM_EPISODES: int = 500_000

# Training
EPSILON: float              = 1.0
EPSILON_DECAY_FACTOR: float = 0.99
EPSILON_MIN: float          = 0.1
ALPHA: float                = 0.1    # Learning Rate
GAMMA: float                = 0.99   # Discount Factor

"""
Scenario: Easy21

Observation space: [player's_cards, dealer's_cards]
Action space: [0 (Stick), 1 (Hit)]
"""
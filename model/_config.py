# Scenario
SCENARIO_OBSERVATIONS: tuple[int]     = (0, 0)
SCENARIO_OBSERVATIONS_NUM: tuple[int] = (10+1, 10+1) #(21+1, 21+1)         # +1 is the offset since array starts at [0]
SCENARIO_ACTIONS: tuple[int]          = (0, 1)
SCENARIO_ACTIONS_NUM: int             = len(SCENARIO_ACTIONS)

# Agent
AGENT_TRAIN: bool = True

# Episode
DEBUG: bool       = False
if AGENT_TRAIN: 
    NUM_EPISODES: int = 1_000_000
else:
    NUM_EPISODES: int = 10_000

# Training
EPSILON: float              = 1.0
EPSILON_DECAY_FACTOR: float = 0.99
EPSILON_MIN: float          = 0.1
ALPHA: float                = 0.1    # Learning Rate
ALPHA_DECAY_FACTOR: float   = 0.999
ALPHA_MIN: float            = 0.01
GAMMA: float                = 0.99   # Discount Factor

# Path
PATH_SAVE_IMAGES: str   = "model/images/"
PATH_SAVE_Q_MATRIX: str = "model/q_matrix/"

"""
Scenario: Easy21

Observation space: [player's_cards, dealer's_cards]
Action space: [0 (Stick), 1 (Hit)]
"""
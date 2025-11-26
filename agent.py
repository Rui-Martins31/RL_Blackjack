import random
import numpy as np

import _config

class Agent:
    def __init__(
            self,
            observation: tuple[int] = _config.SCENARIO_OBSERVATIONS,
            num_observations: tuple[int] = _config.SCENARIO_OBSERVATIONS_NUM,
            actions: tuple[int] = _config.SCENARIO_ACTIONS,
            num_actions: int = _config.SCENARIO_ACTIONS_NUM,
            alpha: float = _config.ALPHA,
            gamma: float = _config.GAMMA, 
            epsilon: float = _config.EPSILON,
            epsilon_decay_factor: float = _config.EPSILON_DECAY_FACTOR,
            epsilon_min: float = _config.EPSILON_MIN
        ):
        
        # Check types
        if not isinstance(observation, tuple): print("[Agent] (__init__) observation is not list."); return
        for idx in range(len(observation)):
            if not isinstance(observation[idx], int): print(f"[Agent] (__init__) observation[{idx}] is not int."); return

        if not isinstance(num_actions, int): print("[Agent] (__init__) num_actions is not int."); return
        #...

        # Info
        self.prev_observation  = observation
        self.num_observations  = num_observations

        self.actions           = actions
        self.num_actions       = num_actions

        # Current state
        # Q-matrix shape: (num_actions, player_sum, dealer_card)
        self.q_matrix          = np.zeros((self.num_actions, *self.num_observations))
        self.is_training: bool = False

        # Training
        self.epsilon: float       = epsilon
        self.epsilon_decay_factor = epsilon_decay_factor
        self.epsilon_min: float   = epsilon_min
        self.lr: float            = alpha
        self.gamma: float         = gamma

    def reset(self, observation: tuple[int]):
        # Reset observations
        self.prev_observation = observation

        # Update epsilon
        self._epsilon_decay()
        
    def update(self, observation: tuple[int], action: int = 0, reward: float = 0.0, done: bool = False):
        # Check types
        if not isinstance(observation, tuple):        print("[Agent] (update) observation is not list.");        return False
        for idx in range(len(observation)):
            if not isinstance(observation[idx], int): print(f"[Agent] (update) observation[{idx}] is not int."); return False

        if not isinstance(reward, float) and not isinstance(reward, int): print("[Agent] (update) reward is not float or int."); return False
        if not isinstance(action, int):   print("[Agent] (update) action is not int.");   return False

        # Update q_matrix
        if self.is_training:
            # Get current Q-value
            current_q = self.q_matrix[(action, *self.prev_observation)]

            # Check terminal state
            if done:
                next_max_q = 0
            else:
                next_max_q = np.amax(self.q_matrix[(slice(None), *observation)])

            # Q-learning update
            self.q_matrix[(action, *self.prev_observation)] = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)

        # Update observations
        self.prev_observation = observation

        return True

    def select(self, greedy: bool = False):
        if (greedy) or (random.random() > self.epsilon):
            # Get Q-values
            q_values = self.q_matrix[(slice(None), *self.prev_observation)]
            return np.argmax(q_values)
        else:
            return random.randint(self.actions[0], self.actions[-1])

    def _epsilon_decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay_factor
        else:
            self.epsilon = self.epsilon_min

    def train(self):
        self.is_training = True

    def test(self):
        self.is_training = False
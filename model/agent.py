import random
import numpy as np
import pickle

import _config

class Agent:
    def __init__(
            self,
            observation: tuple[int] = _config.SCENARIO_OBSERVATIONS,
            num_observations: tuple[int] = _config.SCENARIO_OBSERVATIONS_NUM,
            actions: tuple[int] = _config.SCENARIO_ACTIONS,
            num_actions: int = _config.SCENARIO_ACTIONS_NUM,
            gamma: float = _config.GAMMA,
            N_0: float = _config.N_0,
            monte_carlo: bool = _config.MONTE_CARLO,
            lambda_param: float = 0.0
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

        # Training parameters
        self.monte_carlo: bool    = monte_carlo
        self.gamma: float         = gamma
        self.N_0: float           = N_0
        self.lambda_param: float  = lambda_param

        # Visit counters
        # N(s, a)
        self.state_action_count   = np.zeros((self.num_actions, *self.num_observations))
        # N(s)
        self.state_count          = np.zeros(self.num_observations)

        # Episode trajectory
        # [(state, action, reward), ...]
        self.episode_trajectory: list[tuple[tuple[int], int, float]] = []

        # Eligibility traces
        # E(s, a)
        self.eligibility_traces   = np.zeros((self.num_actions, *self.num_observations))

        # Store previous
        self.prev_action: int = 0

    def reset(self, observation: tuple[int]):
        # Reset observations
        self.prev_observation = observation

        # Clear episode trajectory
        if self.monte_carlo:
            self.episode_trajectory = []

        # Reset eligibility traces
        if not self.monte_carlo and self.lambda_param > 0:
            self.eligibility_traces = np.zeros((self.num_actions, *self.num_observations))
        
    def update(self, observation: tuple[int], action: int = 0, reward: float = 0.0, done: bool = False):
        # Check types
        if not isinstance(observation, tuple):        print("[Agent] (update) observation is not list.");        return False
        for idx in range(len(observation)):
            if not isinstance(observation[idx], int): print(f"[Agent] (update) observation[{idx}] is not int."); return False

        if not isinstance(reward, float) and not isinstance(reward, int): print("[Agent] (update) reward is not float or int."); return False
        if not isinstance(action, int):   print("[Agent] (update) action is not int.");   return False

        if self.is_training:
            if self.monte_carlo:
                # Store trajectory
                self.episode_trajectory.append((self.prev_observation, action, reward))

                # Update Q-values
                if done:
                    self._monte_carlo_update()
            else:
                # TD Learning
                if self.lambda_param > 0:
                    self._sarsa_lambda_update(observation, action, reward, done)
                else:
                    self._td_update(observation, action, reward, done)

        # Update observations
        self.prev_observation = observation

        return True

    def _monte_carlo_update(self):

        G = 0.0  # Return 
        visited_state_actions = set()

        # Process trajectory backwards
        for t in range(len(self.episode_trajectory) - 1, -1, -1):
            state, action, reward = self.episode_trajectory[t]

            # Update return
            G = self.gamma * G + reward

            # First-visit
            state_action = (action, *state)
            if state_action not in visited_state_actions:
                visited_state_actions.add(state_action)

                # Increment visit
                self.state_action_count[state_action] += 1

                # learning rate
                alpha_t = 1.0 / self.state_action_count[state_action]

                # Get current Q-value
                current_q = self.q_matrix[state_action]

                # Monte Carlo update
                self.q_matrix[state_action] = current_q + alpha_t * (G - current_q)

    def _td_update(self, observation: tuple[int], action: int, reward: float, done: bool):

        # Increment visit counter
        self.state_action_count[(action, *self.prev_observation)] += 1

        # learning rate
        alpha_t = 1.0 / self.state_action_count[(action, *self.prev_observation)]

        # Get current Q-value
        current_q = self.q_matrix[(action, *self.prev_observation)]

        # Check terminal state
        if done:
            next_max_q = 0
        else:
            next_max_q = np.amax(self.q_matrix[(slice(None), *observation)])

        # Q-learning update
        self.q_matrix[(action, *self.prev_observation)] = current_q + alpha_t * (reward + self.gamma * next_max_q - current_q)

    def _sarsa_lambda_update(self, observation: tuple[int], action: int, reward: float, done: bool):

        # Next action
        if done:
            next_action = 0
        else:
            self.state_count[observation] += 1
            epsilon_t = self.N_0 / (self.N_0 + self.state_count[observation])

            if random.random() > epsilon_t:
                q_values = self.q_matrix[(slice(None), *observation)]
                next_action = np.argmax(q_values)

            else:
                next_action = random.randint(self.actions[0], self.actions[-1])

        # Increment visit
        self.state_action_count[(action, *self.prev_observation)] += 1

        # Learning rate
        alpha_t = 1.0 / self.state_action_count[(action, *self.prev_observation)]

        # Q-value
        current_q = self.q_matrix[(action, *self.prev_observation)]

        # TD error
        if done:
            next_q = 0
        else:
            next_q = self.q_matrix[(next_action, *observation)]

        td_error = reward + self.gamma * next_q - current_q

        # Update eligibility trace
        self.eligibility_traces[(action, *self.prev_observation)] += 1

        # Update all Q-values
        self.q_matrix += alpha_t * td_error * self.eligibility_traces

        # Decay eligibility traces
        self.eligibility_traces *= self.gamma * self.lambda_param

        # Store next action
        self.prev_action = next_action if not done else 0

    def select(self, greedy: bool = False):
        # Increment state visit
        if self.is_training:
            self.state_count[self.prev_observation] += 1

        # Calculate epsilon
        epsilon_t = self.N_0 / (self.N_0 + self.state_count[self.prev_observation])

        if (greedy) or (random.random() > epsilon_t):
            # Get Q-values
            q_values = self.q_matrix[(slice(None), *self.prev_observation)]
            return np.argmax(q_values)
        
        else:
            return random.randint(self.actions[0], self.actions[-1])

    def train(self):
        self.is_training = True

    def test(self):
        self.is_training = False

    def save_q_matrix(self, filepath: str = "q_matrix.pkl"):
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_matrix, f)
        print(f"[Agent] Q-matrix saved to {filepath}")

    def load_q_matrix(self, filepath: str = "q_matrix.pkl"):
        try:
            with open(filepath, 'rb') as f:
                self.q_matrix = pickle.load(f)
            print(f"[Agent] Q-matrix loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"[Agent] File {filepath} not found.")
            return False
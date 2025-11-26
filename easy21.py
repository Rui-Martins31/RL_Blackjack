import numpy as np
import random

# Globals
STICK: int = 0
HIT: int   = 1

DEALER: int = 0
PLAYER: int = 1

class Easy21:
    def __init__(self):

        # Observations
        self.cards_dealer: list[int] = []   # First card: 1-10
        self.cards_player: list[int] = []   # Total sum: 1-21

        # Actions
        self.actions: list[str] = [STICK, HIT]    # Stick, Hit

        # Probabilities
        self.draw_values: tuple[int] = (1, 10) # If we draw a red card we have (-10, -1) instead
        self.prob_red: float    = 1/3
        self.prob_black: float  = 2/3
    
    def _draw_card(self) -> int:
        card: int = 0
        while (-self.draw_values[0] < card < self.draw_values[0]):
            # CAREFUL!!! We can draw a red card. It should be a value in [-10, -1] U [1, 10]
            card  = random.randint(-self.draw_values[-1], self.draw_values[-1])
        
        return card

    def _dealer_logic(self):
        while (1 < sum(self.cards_dealer) < 17):
            self.cards_dealer.append(self._draw_card())

    def reset(self) -> tuple[int]:
        # Reset
        self.cards_dealer.clear()
        self.cards_player.clear()

        # Draw black cards
        state: tuple[int] = (
            random.randint(*self.draw_values),
            random.randint(*self.draw_values)
        )
        
        # New state
        self.cards_dealer.append(state[DEALER])
        self.cards_player.append(state[PLAYER])

        return state

    def step(self, action: int) -> tuple[tuple[int], bool, float]:
        # State = [7, 18]
        
        # Vars
        terminal: bool        = False
        reward: float         = 0.0

        # Step
        match action:
            case 0: # Stick
                self._dealer_logic()

            case 1: # Hit
                self.cards_player.append(self._draw_card())

                self._dealer_logic()

        # New state
        new_state: tuple[int] = (
            sum(self.cards_dealer),
            sum(self.cards_player)
        )

        # Check results
        terminal     = True
        if (new_state[DEALER] > 21) or (new_state[DEALER] < 1):
            print(f"Dealer busted!")
            reward   = 1.0
            return new_state, terminal, reward
        if (new_state[PLAYER] > 21) or (new_state[PLAYER] < 1):
            print(f"Player busted!")
            reward   = -1.0
            return new_state, terminal, reward

        if (new_state[DEALER] > new_state[PLAYER]):
            print(f"Player LOSES!")
            reward   = -1.0
        elif (new_state[DEALER] < new_state[PLAYER]):
            print(f"Player WINS!")
            reward   = 1.0
        else:
            print(f"DRAW!")
            reward   = 0.0
        
        return new_state, terminal, reward
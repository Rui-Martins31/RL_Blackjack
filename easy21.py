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
        self.prob_red: float   = 1/3
        self.prob_black: float = 2/3

    def update(self, state: tuple[int]):
        self.cards_dealer.append(state[0])
        self.cards_player.append(state[1])

    def reset(self) -> tuple[int]:
        # Reset
        self.cards_dealer.clear()
        self.cards_player.clear()

        # Draw black cards
        state: tuple[int] = ( random.randint(*self.draw_values), random.randint(*self.draw_values) )
        
        # Update
        self.update(state)

        return state

    def step(self, state: tuple[int], action: int) -> tuple[tuple[int], bool, float]:
        # State = [7, 18]
        
        # Vars
        new_state: tuple[int] = state
        terminal: bool        = False
        reward: float         = 0.0

        # Step
        match action:
            case 0: # Stick
                draw_dealer: int = random.randint(*self.draw_values) # CAREFUL!!! We can draw a red card. It should be a value in [-10, -1] U [1, 10]

                new_state[DEALER]    += draw_dealer

            case 1: # Hit
                draw_dealer: int = random.randint(*self.draw_values)
                draw_player: int = random.randint(*self.draw_values)

                new_state[DEALER]    += draw_dealer
                new_state[PLAYER]    += draw_player

        # Check terminal
        if (new_state[DEALER] > 21) or (new_state[DEALER] < 1):
            print(f"Dealer busted!")
        if (new_state[PLAYER] > 21) or (new_state[PLAYER] < 1):
            print(f"Player busted!")
        
        return new_state, terminal, reward
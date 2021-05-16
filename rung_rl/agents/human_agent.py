import random
from .abstract_agent import Agent
from rung_rl.deck import Suit
import torch

class HumanAgent(Agent):
    def __init__(self, id):
        print("You are player", id)
        self.id = id
        self.rewards = 0

    def get_move(self, state):
        action_mask = state.get_action_mask()
        cards = state.cards
        print("Your cards: ", list(zip(cards, list(range(13)))))
        move = int(input("Move: "))
        while action_mask[move] == 0:
            print("Invalid move")
            move = int(input("Move: "))
        return torch.tensor([[move]], dtype=torch.long)

    def reward(self, val, *args):
        self.rewards += val

    def get_rung(self, state, id):
        suits = [suit for suit in Suit]
        cards = state.cards
        print("Available rung suits: ")
        print(suits)
        print("Your cards a.t.m: ", list(zip(cards, list(range(13)))))
        move = int(input("Select Rung: "))
        while move < 1 or move > 4:
            print("Invalid move")
            move = int(input("Select Rung: "))
        return torch.tensor([[move-1]], dtype=torch.long)

    def end(self, win, i):
        if win:
            print("You win")
        else:
            print("You lose")
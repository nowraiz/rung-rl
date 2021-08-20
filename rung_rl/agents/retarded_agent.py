import random
from typing import List

import torch

from .abstract_agent import Agent
from ..deck import Card
from ..state import State


class RetardedAgent(Agent):
    """
    A retarded agent, or as some would say: MOI. 
    It ensures that it always acts sub-optimally unless of course,
    it has to act optimally because that is the only move 
    available
    """

    def __init__(self) -> None:
        self.rewards = 0
        self.last_game_rewards = 0
        self.steps = 0
        self.wins = 0

    def get_move(self, state: State):
        action_mask = state.action_mask
        rung = state.rung
        cards: List[Card] = state.cards[state.player_id]
        # get the playable cards
        moves = []
        playable_cards = []
        for i, card in enumerate(cards):
            if card and action_mask[i] == 1:
                playable_cards.append(card)
                moves.append(i)
        # sort the playable cards
        playable_cards = self.sort_cards(playable_cards)
        # starting from the least cards which is not of rung and play that card
        non_rung_cards = [card for card in playable_cards if card.suit != rung]
        rung_cards = [card for card in playable_cards if card.suit == rung]
        if len(non_rung_cards) == 0:
            # have to play rung
            move = rung_cards[0]
        else:
            move = non_rung_cards[0]

        # find the index at which this card was originally and return that
        for i, card in enumerate(cards):
            if card == move:
                return torch.tensor([[i]], dtype=torch.long)

        assert False

    def get_rung(self, *args):
        return torch.tensor([[random.randint(0, 3)]], dtype=torch.long)

    def reward(self, val, *args):
        self.rewards += val

    def save_obs(self, *args):
        pass

    def end(self, win, *args):
        pass

    def get_wins(self):
        return 0

    def get_rewards(self):
        return self.last_game_rewards

    def reset(self, *args):
        return 0, 0

    def get_steps(self):
        steps = self.steps
        self.steps = 0
        return steps

    def sort_cards(self, cards):
        return sorted(cards, key=lambda card: card.face.value)

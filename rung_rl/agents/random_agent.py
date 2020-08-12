import random

import torch


class RandomAgent():
    def __init__(self, id):
        self.rewards = 0
        self.last_game_rewards = 0

    def get_move(self, cards, hand, stack, rung, num_hand, dominating, last_hand, highest, action_mask, last_dominant):
        choice = [i for i, _ in enumerate(action_mask) if action_mask[i]]
        return torch.tensor([[random.choice(choice)]], dtype=torch.long)

    def reward(self, val, *args):
        self.rewards += val

    def save_obs(self, *args):
        pass

    def end(self, *args):
        self.last_game_rewards = self.rewards
        self.rewards = 0
        pass

    def optimize_model(self, *args):
        pass

    def save_model(self):
        pass

    def get_rewards(self):
        return self.last_game_rewards

import random
from .abstract_agent import Agent
import torch


class RandomAgent(Agent):
    def __init__(self, id):
        self.rewards = 0
        self.last_game_rewards = 0
        self.steps = 0 
        self.wins = 0

    def get_move(self, state):
        action_mask = state.action_mask
        choice = [i for i, _ in enumerate(action_mask) if action_mask[i]]
        self.steps += 1
        return torch.tensor([[random.choice(choice)]], dtype=torch.long)

    def get_rung(self, state, *args):
        return torch.tensor([[random.randint(0,3)]], dtype=torch.long)

    def reward(self, val, *args):
        self.rewards += val

    def save_obs(self, *args):
        pass

    def end(self, win, *args):
        self.wins += int(win)
        self.last_game_rewards = self.rewards
        self.rewards = 0
        pass

    def get_wins(self):
        return 0

    def get_rewards(self):
        return self.last_game_rewards

    def get_steps(self):
        steps = self.steps
        self.steps = 0
        return steps

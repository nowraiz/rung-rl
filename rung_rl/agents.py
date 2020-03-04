import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import rung_rl.utils as utils
from rung_rl.model import Policy
from rung_rl.storage import RolloutStorage
from rung_rl.ppo_algo import PPO

args = {}
args["use_gae"] = True
args["cuda"] = False
args["clip_param"] = 0.2
args["ppo_epoch"] = 4
args["num_mini_batch"] = 1
args["value_loss_coef"] = 0.5
args["entropy_coef"] = 0.02
args["gamma"] = 0.995
args["lr"] = 0.00025
args["eps"] = 1e-5
args["max_grad_norm"] = 0.5
args["gae_lambda"] = 0.95
args["num_steps"] = 14
args["num_processes"] = 1
torch.set_num_threads(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



OBSERVATION_SPACE = 86
ACTIONS = 13
OBSERVATION_SPACE_SHAPE = torch.Size([OBSERVATION_SPACE])
MODEL_PATH = os.getcwd() + "/ppo_rung.pt"


actor_critic = None
try:
    actor_critic = torch.load(MODEL_PATH)
except FileNotFoundError:
    actor_critic = Policy(
        OBSERVATION_SPACE_SHAPE,
        ACTIONS)
    actor_critic.to(device)

policy = PPO(
    actor_critic,
    args["clip_param"],
    args["ppo_epoch"],
    args["num_mini_batch"],
    args["value_loss_coef"],
    args["entropy_coef"],
    lr=args["lr"],
    eps=args["eps"],
    max_grad_norm=args["max_grad_norm"])

def save_policy(i):
    torch.save(actor_critic, MODEL_PATH+str(i))


class PPOAgent():
    def __init__(self, eval=False):
        self.actor_critic = actor_critic
        self.rollouts = RolloutStorage(args["num_steps"], 1, OBSERVATION_SPACE_SHAPE, ACTIONS, actor_critic.recurrent_hidden_state_size)
        self.rollouts.to(device)
        self.cards_seen = [0 for _ in range(52)]
        self.step = 0
        self.rewards = 0
        self.invalid_moves = 0
        self.eval = eval
        self.gamma = args["gamma"]
        self.total_steps = 0
    def get_move(self, cards, hand):
        self.obs = self.get_obs(cards,hand)
        if self.step == 0:
            self.rollouts.obs[0].copy_(self.obs)
        if self.step >= args["num_steps"]:
            self.train()
            self.step = 1
        with torch.no_grad():
            self.value, self.action, self.action_log_prob, self.recurrent_hidden_states = self.actor_critic.act(
                self.rollouts.obs[self.step], self.rollouts.recurrent_hidden_states[self.step],
                self.rollouts.masks[self.step])
        return self.action
    def reward(self, r, invalid = False):
        if invalid:
            self.invalid_moves += 1
        else:
            self.total_steps += 1
        self.rewards += (self.gamma**self.total_steps)*r
        mask = torch.FloatTensor([[0.0]])
        bad_mask = torch.FloatTensor([[0.0]])
        self.rollouts.insert(self.obs, self.recurrent_hidden_states, self.action, self.action_log_prob, self.value, torch.Tensor([r]), mask, bad_mask )
        self.step += 1

    def get_obs(self, cards, hand):
        return torch.Tensor(utils.flatten([card.to_int() if card else (0,0) for card in cards]) + utils.flatten([card.to_int() if card else (0,0) for card in hand]) + [card for card in self.cards_seen])

    def save_obs(self, hand):
        for card in hand:
            if not card:
                break
            idx = card.suit.value * 13 + card.face.value - 2
            self.cards_seen[idx] = 1

    def train(self):
        if self.eval: # do not train if in evaluation mode
            return
        with torch.no_grad():
            next_value = actor_critic.get_value(
                self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                self.rollouts.masks[-1]).detach()
        self.rollouts.compute_returns(next_value, args["use_gae"], args["gamma"],
                                 args["gae_lambda"], False)

        value_loss, action_loss, dist_entropy = policy.update(self.rollouts)
        self.rollouts.after_update()

    def end(self):
        self.train()
        # print(self.rewards)

class RandomAgent():
    def __init__(self):
        self.rewards = 0
    def get_move(self, cards, hand):
        return random.randint(0,12)
    def reward(self, val, invalid=False):
        self.rewards+= val
    def save_obs(self, obs):
        pass
    def end(self):
        pass
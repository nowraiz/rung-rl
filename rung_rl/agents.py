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
from rung_rl.obs import Observation

args = {}
args["use_gae"] = True
args["cuda"] = False
args["clip_param"] = 0.2
args["ppo_epoch"] = 4
args["num_mini_batch"] = 1
args["value_loss_coef"] = 0.5
args["entropy_coef"] = 0.01
args["gamma"] = 0.995
args["lr"] = 0.00025
args["eps"] = 0.2
args["max_grad_norm"] = 0.5
args["gae_lambda"] = 0.95
args["num_steps"] = 13
args["num_processes"] = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



OBSERVATION_SPACE = 303
ACTIONS = 13
OBSERVATION_SPACE_SHAPE = torch.Size([OBSERVATION_SPACE])
MODEL_PATH = os.getcwd() + "/models"
MODEL_NAME = MODEL_PATH + "/vfinal.pt"

actor_critic = None
try:
    print("Loading the network from file...")
    actor_critic = torch.load(MODEL_NAME)
except FileNotFoundError:
    print("File not found. Creating a new network...")
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
    torch.save(actor_critic, MODEL_PATH+"/v"+str(i) +".pt")

def update_lr(i,n):
    utils.update_linear_schedule(policy.optimizer, i, n,args["lr"])

class PPOAgent():
    def __init__(self,id, eval=False):
        self.actor_critic = actor_critic
        self.rollouts = RolloutStorage(args["num_steps"], 1, OBSERVATION_SPACE_SHAPE, ACTIONS, actor_critic.recurrent_hidden_state_size)
        self.rollouts.to(device)
        self.cards_seen = [0 for _ in range(52)]
        self.step = 0
        self.rewards = 0
        self.invalid_moves = 0
        self.eval = eval
        self.id = id
    def get_move(self, cards, hand, stack, rung, num_hand, dominating,action_mask):
        self.obs = self.get_obs(cards,hand, stack, rung, num_hand, dominating)
        if self.step == 0:
            self.rollouts.obs[0].copy_(self.obs)
            self.rollouts.to(device)
        with torch.no_grad():
            self.value, self.action, self.action_log_prob, self.recurrent_hidden_states = self.actor_critic.act(
                self.rollouts.obs[self.step], self.rollouts.recurrent_hidden_states[self.step],
                self.rollouts.masks[self.step], action_mask)
        # add this card to the played card
        self.save_card(cards[self.action])
        return self.action
    
    def reward(self, r, done = False, invalid = False):
        if invalid:
            self.invalid_moves += 1
        self.rewards += r
        mask = torch.FloatTensor([[0.0]] if done else [[1.0]]).to(device)
        bad_mask = torch.FloatTensor([[1.0]]).to(device)
        self.rollouts.insert(self.obs, self.recurrent_hidden_states, self.action, self.action_log_prob, self.value, torch.Tensor([r]).to(device), mask, bad_mask )
        self.step += 1

    def get_obs(self, cards, hand,stack,rung, num_hand, dominating):
        obs = Observation(cards, hand, stack, rung, num_hand, dominating, self.id)
        return torch.Tensor(obs.get()).to(device)

    def save_obs(self, hand):
        return # temp
        for card in hand:
            if not card:
                break
            self.save_card(card)
    def save_card(self, card):
        idx = (card.suit.value - 1) * 13 + card.face.value - 2
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

    def end(self):
        self.train()
        # print(self.rewards)

class RandomAgent():
    def __init__(self, id):
        self.rewards = 0
    def get_move(self, cards, hand, stack, rung, num_hand, dominating,action_mask):
        choice = [i for i, _ in enumerate(action_mask) if action_mask[i]]
        return random.choice(choice)
    def reward(self, val, *args):
        self.rewards+= val
    def save_obs(self, *args):
        pass
    def end(self, *args):
        pass


class HumanAgent():
    def __init__(self, id):
        print("You are player", id)
        self.id = id
        self.rewards = 0
    def get_move(self, cards, hand, stack, rung, num_hand, dominating, action_mask):
        print("Your cards: ", list(zip(cards, list(range(13)))))
        move = int(input("Move: "))
        while action_mask[move] == 0:
            print("Invalid move")
            move = int(input("Move: "))
        return move
    def reward(self, val, *args):
        self.rewards += val
    def save_obs(self, *args):
        pass
    def end(self, *args):
        pass
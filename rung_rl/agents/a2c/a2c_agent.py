import os

import torch
import random
import math
import torch.optim as optim
import torch.nn.functional as F
from ..dqn.dqn_network import DQNNetwork
# from .rung_network import RungNetwork
# from .replay_memory import ReplayMemory, Transition, ActionMemory, StateAction
from ...obs import Observation
import sys

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
gpu = torch.device("cuda")

# BATCH_SIZE = 64
GAMMA = 0.999
# EPS_START = 0.3
# EPS_END = 0.05
# EPS_DECAY = 1000000
# TARGET_UPDATE = 1000
# MIN_BUFFER_SIZE = 1000
# RUNG_BATCH_SIZE = 64
NUM_ACTIONS = 13
INPUTS = 1418
LEARNING_STARTS = 1000
MODEL_PATH = os.getcwd() + "/models/a2c"
LR = 5e-5



class A2CAgent:
    def __init__(self, train=True):
        # self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        # self.EPS_START = EPS_START
        # self.EPS_END = EPS_END
        # self.EPS_DECAY = EPS_DECAY
        # self.TARGET_UPDATE = TARGET_UPDATE
        # self.RUNG_BATCH_SIZE = RUNG_BATCH_SIZE
        self.num_actions = NUM_ACTIONS
        self.steps = 0 # the total steps taken by the agent
        self.rewards = [] # rewards acheived at each step
        self.log_probs = [] # log probs of action taken at each step
        # self.actions = [] # the actions taken at each step
        # self.states = [] # the states (does not really matter)
        self.values = [] # the values predicted by the critic
        self.dones = []

        self.actor = DQNNetwork(INPUTS, NUM_ACTIONS).to(device)
        self.critic = DQNNetwork(INPUTS, 1).to(device) # there is only one output which is the value of the state
        # self.target_net = DQNNetwork(INPUTS, NUM_ACTIONS).to(device).eval()
        # self.rung_net = RungNetwork(85, 4).to(device)
        # self.rung_optimizer = optim.Adam(self.rung_net.parameters(),lr=1e-4)
        # self.rung_memory = ReplayMemory(100000)
        # self.average_policy = DQNNetwork(INPUTS, NUM_ACTIONS).to(device)
        # self.policy_optimizer = optim.RMSprop(self.average_policy.parameters())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)

        # self.action_memory = ActionMemory(1000000)
        # self.last_actions = [None, None, None, None]
        # self.last_rewards = [0, 0, 0, 0]
        # self.last_states = [None, None, None, None]
        self.total_reward = 0
        self.train = train
        self.wins = 0
        # self.rung_selected = [None, None, None, None]
        # self.rung_state = [None, None, None, None]
        self.deterministic = False
        self.steps = 0
        self.eval = False
        # self.last_ga?me_reward = 0
        # self.cards_seen_index = 0
        self.load_model()

    def get_rung(self, state, player):
        return torch.tensor([random.randint(0,3)]) # return a random rung for now
        # state = self.get_rung_obs(state)
        # self.rung_state[player] = state
        # self.rung_selected[player] = self.select_rung(self.rung_state[player])
        # return self.rung_selected[player]

    def select_action(self, state, action_mask):
        probs = self.actor(state)
        mask = self.create_action_mask_tensor(action_mask)
        sm = torch.nn.Softmax(1)
        # print(action_mask)
        # print(sm(probs))
        probs = probs + mask
        dist = torch.distributions.Categorical(probs=sm(probs))
        action = dist.sample()
        # return sm(probs).max(1)[1], dist.log_prob(action)
        return action, dist.log_prob(action)

    def reward(self, r, player, done=False):
        # self.last_rewards[player] = torch.tensor([[r]], dtype=torch.float).to(device)
        self.total_reward += r
        self.rewards.append(r)
        if done:
            self.dones.append(0)
        else:
            self.dones.append(1)

    def get_value(self, state):
        out = self.critic(state)
        return out.view(1,1)

    def get_move(self, state):
        player = state.player_id
        action_mask = state.get_action_mask()
        state = self.get_obs(state)
        value = self.get_value(state)
        action, log_prob = self.select_action(state, action_mask)
        self.log_probs.append(log_prob)
        # print(value)
        self.values.append(value)
        self.steps += 1
        # self.last_states[player] = state
        # self.last_actions[player] = self.select_action(state, action_mask, player)

        return action
    
    def create_action_mask_tensor(self, mask):
        return torch.tensor([[0 if m == 1 else float("-inf") for m in mask]], device=device)

    def get_obs(self, state):
        obs = state.get_obs()
        return torch.tensor([obs.get()], dtype=torch.float).to(device)

    def calculate_returns(self):
        returns = 0
        for i in range(len(self.rewards)):
            returns = self.rewards[i] + GAMMA*returns*self.dones[i]
            self.rewards[i] = returns

    def optimize_model(self):
        if self.eval:
            return
        self.calculate_returns()
        loss_actor = self.optimize_actor()
        loss_critic = self.optimize_critic()
        # print(loss_critic)
        print("actor: {:.5f} critic: {:.5f}".format(loss_actor, loss_critic), end=" - ")

        # clear the trajectory
        self.rewards = []
        # self.steps = 0
        self.log_probs = []
        self.values = []
        self.dones = []

    def optimize_actor(self):
        # print(self.log_probs)
        # print(self.rewards)
        advantages = torch.tensor(self.rewards) - torch.cat(self.values, 1)
        log_probs = torch.cat(self.log_probs, 0)
        # print(self.log_probs)
        # print(log_probs)
        # print(advantages)
        actor_loss = (-1 * log_probs)*advantages
        # print(actor_loss)
        # print()
        # print(actor_loss)
        actor_loss_mean = torch.mean(actor_loss)
        # print(actor_loss_mean)
        self.actor_optimizer.zero_grad()
        actor_loss_mean.backward(retain_graph=True)
        self.actor_optimizer.step()
        return actor_loss_mean.item()

    def optimize_critic(self):
        advantages = torch.tensor(self.rewards) - torch.cat(self.values, 1)
        critic_loss = 0.5 * torch.mean(torch.square(advantages))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss.item()
        # self.optimizer.zero_grad()
        # loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # self.optimizer.step()
        # if self.steps_done % 1300 == 0:
        # print(loss.item())
        # return loss.item()

    def end(self, win, player):
        self.wins += win
        self.total_reward = 0
        # self.memory.push(self.last_state, self.last_action, None, self.last_reward)
        # self.last_game_reward = self.game_reward
        # self.game_reward = 0
        # self.last_state = None
        # self.last_action = None
        # self.last_reward = None
        # self.cards_seen = [None for _ in range(52)]
        # self.cards_seen_index = 0
        # do nothing at the end of the game
        pass

    def reset(self):
        wins = self.wins
        self.wins = 0
        return wins

    def save_model(self, i="final"):
        torch.save(self.actor.state_dict(), self.model_path("actor"))
        torch.save(self.critic.state_dict(), self.model_path("critic"))
        # torch.save(self.average_policy.state_dict(), self.average_model_path(i))

    def load_model(self, i="final"):
        try:
            state_dict = torch.load(self.model_path("actor"))
            # self.policy_net.load_state_dict(state_dict)
            self.actor.load_state_dict(state_dict)
            state_dict = torch.load(self.model_path("critic"))
            self.critic.load_state_dict(state_dict)
            # state_dict = torch.load(self.average_model_path(i))
            # self.average_policy.load_state_dict(state_dict)
        except FileNotFoundError:
            print("File not found. Creating a new network...")

    def model_path(self, model_name, i="final"):
        return "{}/model_{}_{}".format(MODEL_PATH, model_name, i)

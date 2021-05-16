import os

import torch
import random
import math
import torch.optim as optim
import torch.nn.functional as F
from ..dqn.dqn_network import DQNNetwork
from .a2c_network import A2CNetwork
# from .rung_network import RungNetwork
# from .replay_memory import ReplayMemory, Transition, ActionMemory, StateAction
from ...obs import Observation
import sys

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
gpu = torch.device("cuda")

# BATCH_SIZE = 64
GAMMA = 0.99
# EPS_START = 0.3
# EPS_END = 0.05
# EPS_DECAY = 1000000
# TARGET_UPDATE = 1000
# MIN_BUFFER_SIZE = 1000
# RUNG_BATCH_SIZE = 64
NUM_ACTIONS = 13
INPUTS = 1482
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
        self.steps = [0, 0, 0, 0] # the total steps taken by the agent
        self.rewards = [[], [], [], []] # rewards acheived at each step
        self.log_probs = [[], [], [], []] # log probs of action taken at each step
        self.entropies = [[], [], [], []] # entropy of each distribution produced
        # self.actions = [] # the actions taken at each step
        # self.states = [] # the states (does not really matter)
        self.values = [[], [], [], []] # the values predicted by the critic
        self.values_tensor = [None, None, None, None]
        self.dones = [[], [], [], []]

        self.actor_critic = A2CNetwork(INPUTS, 128, NUM_ACTIONS).to(device)
        # self.critic = A2CNetwork(INPUTS, 128, 1).to(device) # there is only one output which is the value of the state
        # self.target_net = DQNNetwork(INPUTS, NUM_ACTIONS).to(device).eval()
        # self.rung_net = RungNetwork(85, 4).to(device)
        # self.rung_optimizer = optim.Adam(self.rung_net.parameters(),lr=1e-4)
        # self.rung_memory = ReplayMemory(100000)
        # self.average_policy = DQNNetwork(INPUTS, NUM_ACTIONS).to(device)
        # self.policy_optimizer = optim.RMSprop(self.average_policy.parameters())
        self.actor_optimizer = optim.Adam(self.actor_critic.parameters(), lr=LR)
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR)

        # self.action_memory = ActionMemory(1000000)
        # self.last_actions = [None, None, None, None]
        # self.last_rewards = [0, 0, 0, 0]
        # self.last_states = [None, None, None, None]
        self.total_reward = [0, 0, 0, 0]
        self.train = train
        self.wins = [0, 0, 0, 0]
        # self.rung_selected = [None, None, None, None]
        # self.rung_state = [None, None, None, None]
        self.deterministic = False
        self.steps = [0, 0, 0, 0]
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
        raw_probs, value = self.actor_critic(state)
        mask = self.create_action_mask_tensor(action_mask)
        # sm = torch.nn.Softmax(-1)
        # print(sm(raw_probs))
        # print(action_mask)
        probs = raw_probs + mask
        # print(action_mask)
        # print(probs)
        # log_probs = torch.log(probs)
        # log_prob = log_probs
        # print(action_mask)
        # print(probs)
        # raw_probs = raw_probs.detach() + mask
        dist = torch.distributions.Categorical(logits=probs)
        if self.eval:
            # print(dist.probs)
            action = dist.probs.max(1)[1]
            # print(action)
        else:
            action = dist.sample()
        # print(log_probs)
        # return sm(probs).max(1)[1], dist.log_prob(action)
        return action, dist.log_prob(action), dist.entropy(), value

    def reward(self, r, player, done=False):
        # self.last_rewards[player] = torch.tensor([[r]], dtype=torch.float).to(device)
        self.total_reward[player] += r
        self.rewards[player].append(r)
        if done:
            self.dones[player].append(0)
        else:
            self.dones[player].append(1)

    def get_value(self, state):
        out = self.critic(state)
        return out.view(1,1)

    def get_move(self, state):
        player = state.player_id
        action_mask = state.get_action_mask()
        state = self.get_obs(state)
        # value = self.get_value(state)
        action, log_prob, _, value = self.select_action(state, action_mask)
        # print(value)
        self.log_probs[player].append(log_prob)
        # print(value)
        self.values[player].append(value)
        self.steps[player] += 1
        # self.last_states[player] = state
        # self.last_actions[player] = self.select_action(state, action_mask, player)

        return action
    
    def create_action_mask_tensor(self, mask):
        return torch.tensor([[0 if m else -1e8 for m in mask]], dtype=torch.float, device=device)

    def get_obs(self, state):
        obs = state.get_obs()
        return torch.tensor([obs.get()], dtype=torch.float).to(device)

    def calculate_returns(self, player):
        # if len(self.rewards[player]) == 1:
            # 1 step a2c
            # future_returns = 
        # print(self.rewards[player])
        returns = 0
        for i in range(len(self.rewards[player])-1, -1, -1):
            returns = self.rewards[player][i] + GAMMA*returns*self.dones[player][i]
            self.rewards[player][i] = returns

    def optimize_model(self):
        if self.eval:
            return
        for player in range(4):
            self.calculate_returns(player)
            self.values_tensor[player] = torch.cat(self.values[player], 1)
        loss_actor_critic = 0
        # loss_critic = 0
        self.actor_optimizer.zero_grad()
        # self.critic_optimizer.zero_grad()
        for player in range(4):
            loss_actor_critic += self.optimize_actor_critic(player)
            # loss_critic += self.optimize_critic(player)
        # loss_actor /= 4
        # loss_critic /= 4
        
        for param in self.actor_critic.parameters():
            param.grad.data.clamp_(-0.5, 0.5)
        # for param in self.critic.parameters():
            # param.grad.data.clamp_(-0.5, 0.5)

        
        self.actor_optimizer.step()
        # self.critic_optimizer.step()
        self.actor_optimizer.zero_grad()
        # self.critic_optimizer.zero_grad()
        # print(loss_critic)
        print("Loss: {:.5f}".format(loss_actor_critic), end=" - ")

        self.clear_trajectory()
        

    def optimize_actor_critic(self, player):
        # print(self.log_probs)
        # print(self.rewards)
        advantages = torch.tensor(self.rewards[player]) - self.values_tensor[player]
        log_probs = torch.cat(self.log_probs[player], 0)
        # print(self.log_probs)
        # print(log_probs)
        # print(advantages)
        actor_loss = (-1 * log_probs)*advantages
        # print(actor_loss)
        # print()
        # print(actor_loss)
        actor_loss_mean = torch.mean(actor_loss) / 4 # averaging across 4 workers (players)
        critic_loss = torch.mean(torch.square(advantages)) / 4
        loss = actor_loss_mean + critic_loss
        # print(actor_loss_mean)
        # self.actor_optimizer.zero_grad()
        loss.backward()
        # self.actor_optimizer.step()
        return loss.item()

    def optimize_critic(self, player):
        advantages = torch.tensor(self.rewards[player]) - self.values_tensor[player]
        critic_loss = torch.mean(torch.square(advantages)) / 4

        # self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # self.critic_optimizer.step()
        return critic_loss.item()
        # self.optimizer.zero_grad()
        # loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # self.optimizer.step()
        # if self.steps_done % 1300 == 0:
        # print(loss.item())
        # return loss.item()

    def clear_trajectory(self):
        # clear the trajectory
        self.rewards = [[], [], [], []]
        # self.steps = 0
        self.log_probs = [[], [], [], []]
        self.values = [[], [], [], []]
        self.dones = [[], [], [], []]
        self.values_tensor = [None, None, None, None]
        

    def end(self, win, player):
        self.wins[player] += win
        # self.total_reward = [0, 0, 0, 0]
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

    def reset(self, player):
        wins = self.wins[player]
        reward = self.total_reward[player]
        self.wins[player] = 0
        self.total_reward[player] = 0
        return wins, reward

    def save_model(self, i="final"):
        torch.save(self.actor_critic.state_dict(), self.model_path("actor_critic"))
        # torch.save(self.critic.state_dict(), self.model_path("critic"))
        # torch.save(self.average_policy.state_dict(), self.average_model_path(i))

    def load_model(self, i="final"):
        try:
            state_dict = torch.load(self.model_path("actor_critic"))
            # self.policy_net.load_state_dict(state_dict)
            self.actor_critic.load_state_dict(state_dict)
            # state_dict = torch.load(self.model_path("critic"))
            # self.critic.load_state_dict(state_dict)
            # state_dict = torch.load(self.average_model_path(i))
            # self.average_policy.load_state_dict(state_dict)
        except FileNotFoundError:
            print("File not found. Creating a new network...")

    def model_path(self, model_name, i="final"):
        return "{}/model_{}_{}".format(MODEL_PATH, model_name, i)

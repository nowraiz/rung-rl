import os

import torch
import random
import torch.nn as nn
import math
import torch.optim as optim
import torch.nn.functional as F
from .nfsp_network import NFSPNetwork
from .replay_memory import ReplayMemory, Transition, ActionMemory, StateAction
from ...obs import Observation
import numpy as np
import sys

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
gpu = torch.device("cuda")

BATCH_SIZE = 64
ANTICIPATION = 0.1
TRAIN_EVERY = 13
GAMMA = 0.999
EPS_START = 0.06
EPS_END = 0
UPDATE_EVERY = 64
EPS_DECAY = 1000000
TARGET_UPDATE = 500
MIN_BUFFER_SIZE = 1000
NUM_ACTIONS = 13
INPUTS = 1418
LEARNING_STARTS = 1
MODEL_PATH = os.getcwd() + "/models/"
LR = 5e-5
Q_LR = 0.00001
S_LR = 0.00005


class NFSPAgent:
    def __init__(self, eval=False):
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE
        self.num_actions = NUM_ACTIONS
        self.steps = 0
        self.policy_net = NFSPNetwork(INPUTS, NUM_ACTIONS).to(device)
        self.target_net = NFSPNetwork(INPUTS, NUM_ACTIONS).to(device).eval()
        self.average_policy_net = NFSPNetwork(INPUTS, NUM_ACTIONS).to(device)
        # self.policy_optimizer = optim.RMSprop(self.average_policy.parameters())
        self.mode = None # the mode the agent is following for the episode 0 = Best-response 1 = Average
        self.sl_optimizer = optim.Adam(self.average_policy_net.parameters(), lr=S_LR)
        self.rl_optimizer = optim.Adam(self.policy_net.parameters(), lr=Q_LR)
        self.memory = ReplayMemory(100000)
        self.action_memory = ActionMemory(1000000)
        self.epsilons = np.linspace(self.EPS_START, self.EPS_END, self.EPS_DECAY)
        self.last_action = None
        self.last_reward = None
        self.last_state = None
        self.total_reward = 0
        self.wins = 0
        self.steps = 0
        self.eval = False
        self.init_average_policy_net()
        self.sample_episode_policy()
        # self.last_ga?me_reward = 0
        # self.cards_seen_index = 0
        self.load_model()
    def sample_episode_policy(self):
        toss = random.random()
        if toss < ANTICIPATION:
            self.mode = 0 # best response
        else:
            self.mode = 1 # average policy

    def init_average_policy_net(self):
        # xavier init
        for p in self.average_policy_net.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)


    def select_action(self, state, action_mask, player):
        if self.mode == 1:
            # average policy following
            # temp_state = torch.rand((1,1199))
            output = self.average_policy_net(state)
            # print("--------------")
            # print(state.shape)
            # print(output)
            # print(action_mask)
            sm = torch.nn.Softmax(1)
            # print(softmax(output))
            mask = torch.tensor([[0 if m == 1 else float("-inf") for m in action_mask]], device=device)
            output = output + mask
            categorical = torch.distributions.Categorical(sm(output))
            # print(output)
            # out = self.policy_net(state)
            # print(out)
            # sys.exit()
            # print("average move")
            if self.eval:
                return sm(output).max(1)[1], False
            else:
                return categorical.sample(), False


        sample = random.random()
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        # math.exp(-1. * self.steps_done[player] / EPS_DECAY)
        eps_threshold = self.EPS_END
        if self.steps < len(self.epsilons):
            eps_threshold = self.epsilons[self.steps]

        if sample > eps_threshold:
            with torch.no_grad():
                out = self.policy_net(state)
                mask = torch.tensor([[0 if m == 1 else float("-inf") for m in action_mask]], device=device)
                # print("------")
                # print(out)
                # print(action_mask)
                out = out + mask
                s = torch.nn.Softmax(1)
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print("value move")
                # print(out)
                # print(out)
                # return torch.distributions.Categorical(s(out)).sample(), True
                # print(out.max(1)[0])
                # print(out.max(1)[1])
                # print(torch.max(out,1)[0])
                # print("-----")
                return out.max(1)[1].view(1, 1), True
        else:
            # print("random")
            choice = [i for i, _ in enumerate(action_mask) if action_mask[i]]
            return torch.tensor([[random.choice(choice)]], device=device, dtype=torch.long), True

    def reward(self, r, player, done=False):
        self.last_reward = torch.tensor([[r]], dtype=torch.float).to(device)
        self.total_reward += r
        if done and not self.eval:
            self.memory.push(self.last_state, self.last_action, None, self.last_reward, None)
            self.last_state = None

    def create_action_mask_tensor(self, mask):
        return torch.tensor([[0 if m == 1 else float("-inf") for m in mask]], device=device)


    def get_move(self, state):
        player = state.player_id
        action_mask = state.get_action_mask()
        state = self.get_obs(state)
        if self.last_state is not None and not self.eval:
            self.memory.push(self.last_state, self.last_action, state, self.last_reward,
                            self.create_action_mask_tensor(action_mask))
        
        self.last_state = state
        self.last_action, following_policy = self.select_action(state, action_mask, player)
        if following_policy and not self.eval:
            self.action_memory.push_with_sampling(state, self.last_action)

        if not self.eval:
            self.steps += 1
        if (self.steps % UPDATE_EVERY == 0):
            self.optimize_model()
            # self.optimize_model()

        if ((self.steps / UPDATE_EVERY) % TARGET_UPDATE == 0 ):
            self.mirror_models()
        # if (self.steps % UPDATE_EVERY == 0):
        #     self.optimize_model()
        
        # if ((self.steps / BATCH_SIZE) % TARGET_UPDATE == 0):
        #     self.mirror_models()
        return self.last_action

    def get_obs(self, state):
        obs = state.get_obs()
        return torch.tensor([obs.get()], dtype=torch.float).to(device)

    def optimize_average_policy(self):
        if len(self.action_memory) < self.BATCH_SIZE or len(self.action_memory) < MIN_BUFFER_SIZE:
            return 0

        actions = self.action_memory.sample(self.BATCH_SIZE)
        batch = StateAction(*zip(*actions))
        actions = torch.cat(batch.action, 1)
        actions = actions.squeeze(0)
        state = torch.cat(batch.state, 0)
        # print(state)
        expected = self.average_policy_net(state)
        # print(actions)
        ce_loss = torch.nn.CrossEntropyLoss()
        loss = ce_loss(expected, actions)
        self.sl_optimizer.zero_grad()
        loss.backward()
        for param in self.average_policy_net.parameters():
            param.grad.data.clamp_(-10, 10)
        self.sl_optimizer.step()
        return loss.item()


    def optimize_model(self):
        if self.eval:
            return
        # print("Optimizing Model...")
        loss_value = self.optimize_value_model()
        loss_avg = self.optimize_average_policy()
        # print("loss_value: {}".format(loss_value))
        print("loss_value: {}, loss_avg: {}".format(loss_value, loss_avg))

    def optimize_value_model(self):
        if len(self.memory) < self.BATCH_SIZE or len(self.memory) < MIN_BUFFER_SIZE:
            return 0

        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)
        action_masks = torch.cat([mask for mask in batch.action_mask if mask is not None])
        # print(action_batch)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # print(action_masks)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # print(state_action_values)
        # Compute V(s_{t+1}) for all next states.
        best_actions = self.policy_net(non_final_next_states) + action_masks # filter out invalid actions
        # print(best_actions)
        best_actions = best_actions.max(1)[1].unsqueeze(1)
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        # print(self.target_net(non_final_next_states).gather(1, best_actions).squeeze(1))
        # print(self.target_net(non_final_next_states).max(1)[0])
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, best_actions).squeeze(1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # print(loss)

        # Optimize the model
        # print(loss.item())
        self.rl_optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-10, 10)
        self.rl_optimizer.step()
        # if self.steps_done % 1300 == 0:
        # print(loss.item())
        return loss.item()

    def end(self, win):
        self.wins += win
        self.total_reward = 0
        pass

    def reset(self):
        wins = self.wins
        self.wins = 0
        return wins

    def save_model(self, i):
        torch.save(self.policy_net.state_dict(), self.model_path(i))
        torch.save(self.average_policy_net.state_dict(), self.average_model_path(i))

    def load_model(self, i="final"):
        try:
            state_dict = torch.load(self.model_path(i))
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            state_dict = torch.load(self.average_model_path(i))
            self.average_policy_net.load_state_dict(state_dict)
        except FileNotFoundError:
            print("File not found. Creating a new network...")

    def model_path(self, i):
        return "{}/model_{}_{}".format(MODEL_PATH, "dqn", i)

    def average_model_path(self, i):
        return "{}/model_{}_{}".format(MODEL_PATH, "avg", i)

    def mirror_models(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

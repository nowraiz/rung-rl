import os

import torch
import random
import math
import torch.optim as optim
import torch.nn.functional as F
from .dqn_network import DQNNetwork
from .rung_network import RungNetwork
from .replay_memory import ReplayMemory, Transition, ActionMemory, StateAction
from ...obs import Observation
import sys

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
gpu = torch.device("cuda")

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.3
EPS_END = 0.05
EPS_DECAY = 1000000
TARGET_UPDATE = 1000
MIN_BUFFER_SIZE = 1000
RUNG_BATCH_SIZE = 64
NUM_ACTIONS = 13
INPUTS = 1418
LEARNING_STARTS = 1000
MODEL_PATH = os.getcwd() + "/models/"
LR = 5e-5



class DQNAgent:
    def __init__(self, train=True):
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE
        self.RUNG_BATCH_SIZE = RUNG_BATCH_SIZE
        self.num_actions = NUM_ACTIONS
        self.steps_done = [0, 0, 0, 0]
        self.policy_net = DQNNetwork(INPUTS, NUM_ACTIONS).to(device)
        self.target_net = DQNNetwork(INPUTS, NUM_ACTIONS).to(device).eval()
        self.rung_net = RungNetwork(85, 4).to(device)
        self.rung_optimizer = optim.Adam(self.rung_net.parameters(),lr=1e-4)
        self.rung_memory = ReplayMemory(100000)
        # self.average_policy = DQNNetwork(INPUTS, NUM_ACTIONS).to(device)
        # self.policy_optimizer = optim.RMSprop(self.average_policy.parameters())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(100000)
        # self.action_memory = ActionMemory(1000000)
        self.last_actions = [None, None, None, None]
        self.last_rewards = [0, 0, 0, 0]
        self.last_states = [None, None, None, None]
        self.total_reward = 0
        self.train = train
        self.wins = 0
        self.rung_selected = [None, None, None, None]
        self.rung_state = [None, None, None, None]
        self.deterministic = False
        self.steps = 0
        self.eval = False
        # self.last_ga?me_reward = 0
        # self.cards_seen_index = 0
        self.load_model()

    def select_rung(self, rung_state):
        with torch.no_grad():
            out = self.rung_net(rung_state)
            # print(out)
            # sys.exit()
            # s = torch.nn.Softmax(1)
            # return torch.distributions.Categorical(s(out)).sample()
            return out.max(1)[1].view(1, 1)

    def get_rung(self, state, player):
        state = self.get_rung_obs(state)
        self.rung_state[player] = state
        self.rung_selected[player] = self.select_rung(self.rung_state[player])
        return self.rung_selected[player]

    def select_action(self, state, action_mask, player):
        sample = random.random()
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        # math.exp(-1. * self.steps_done[player] / EPS_DECAY)
        eps_threshold = ((EPS_END) + (EPS_START - EPS_END)) / (self.steps_done[player]+ 1)
        self.steps_done[player] += 1
        if sample > eps_threshold or self.eval:
            with torch.no_grad():
                out = self.policy_net(state)
                # print(out)
                # print(action_mask)
                mask = torch.tensor([[0 if m == 1 else float("-inf") for m in action_mask]], device=device)
                out = out + mask
                # print(out)
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print("value move")
                # print(out)
                # return torch.distributions.Categorical(s(out)).sample(), True
                return out.max(1)[1].view(1, 1)
        else:
            # print("random")
            choice = [i for i, _ in enumerate(action_mask) if action_mask[i]]
            return torch.tensor([[random.choice(choice)]], device=device, dtype=torch.long)

    def reward(self, r, player, done=False):
        self.last_rewards[player] = torch.tensor([[r]], dtype=torch.float).to(device)
        self.total_reward += r
        if done:
            self.memory.push(self.last_states[player], self.last_actions[player], None, self.last_rewards[player], None)
            self.last_states[player] = None

    def get_move(self, state):
        player = state.player_id
        action_mask = state.get_action_mask()
        state = self.get_obs(state)
        if self.last_states[player] is not None and not self.eval:
            self.memory.push(self.last_states[player], self.last_actions[player], state,
                             self.last_rewards[player], self.create_action_mask_tensor(action_mask))
        self.last_states[player] = state
        self.last_actions[player] = self.select_action(state, action_mask, player)

        return self.last_actions[player]
    
    def create_action_mask_tensor(self, mask):
        return torch.tensor([[0 if m == 1 else float("-inf") for m in mask]], device=device)

    def get_rung_obs(self, state):
        obs = state.get_obs()
        return torch.tensor([obs.get_rung()], dtype=torch.float).to(device)

    def get_obs(self, state):
        obs = state.get_obs()
        return torch.tensor([obs.get()], dtype=torch.float).to(device)

    def optimize_rung_network(self):
        if self.eval or len(self.rung_memory) < self.RUNG_BATCH_SIZE:
            return 0

        sampled = self.rung_memory.sample(self.RUNG_BATCH_SIZE)
        batch = Transition(*zip(*sampled))
        # print(batch.action)
        actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], dtype=torch.float, device=device), batch.reward)))
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        state_action_values = self.rung_net(state_batch).gather(1, action_batch)
        loss = F.mse_loss(state_action_values, reward_batch.unsqueeze(1))
        self.rung_optimizer.zero_grad()
        loss.backward()
        self.rung_optimizer.step()
        return loss.item()

    def optimize_average_policy(self):
        if len(self.action_memory) < self.BATCH_SIZE:
            return 0

        actions = self.action_memory.sample(self.BATCH_SIZE)
        batch = StateAction(*zip(*actions))
        actions = torch.cat(batch.action, 1)
        actions = actions.squeeze(0)
        state = torch.cat(batch.state, 0)
        # print(state)
        expected = self.average_policy(state)
        # print(actions)
        loss = F.cross_entropy(expected, actions)
        self.policy_optimizer.zero_grad()
        loss.backward()
        # for param in self.average_policy.parameters():
        #     param.grad.data.clamp_(-10, 10)
        self.policy_optimizer.step()
        return loss.item()


    def optimize_model(self):
        if self.eval:
            return
        # print("Optimizing Model...")
        loss_value = self.optimize_value_model()
        # loss_avg = self.optimize_average_policy()
        print("loss_value: {}".format(loss_value), end=" -- ")
        # print("loss_value: {}, loss_avg: {}".format(loss_value, loss_avg))

    def optimize_value_model(self):
        if len(self.memory) < self.BATCH_SIZE or len(self.memory) < MIN_BUFFER_SIZE or self.eval:
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
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # if self.steps_done % 1300 == 0:
        # print(loss.item())
        return loss.item()

    def end(self, win, player):
        if not self.eval and self.rung_state[player] is not None:
            self.rung_memory.push(self.rung_state[player], self.rung_selected[player], None, 1 if win else -1, None)
        self.rung_selected[player], self.rung_state[player] = None, None
        self.wins += win
        # print(self.total_reward)
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

    def save_model(self, i):
        torch.save(self.policy_net.state_dict(), self.model_path(i))
        torch.save(self.rung_net.state_dict(), self.rung_model_path(i))
        # torch.save(self.average_policy.state_dict(), self.average_model_path(i))

    def load_model(self, i="final"):
        try:
            state_dict = torch.load(self.model_path(i))
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            state_dict = torch.load(self.rung_model_path(i))
            self.rung_net.load_state_dict(state_dict)
            # state_dict = torch.load(self.average_model_path(i))
            # self.average_policy.load_state_dict(state_dict)
        except FileNotFoundError:
            print("File not found. Creating a new network...")

    def model_path(self, i):
        return "{}/model_{}_{}".format(MODEL_PATH, "dqn", i)

    def rung_model_path(self, i):
        return "{}/model_{}_{}".format(MODEL_PATH, "rung", i)

    def average_model_path(self, i):
        return "{}/model_{}_{}".format(MODEL_PATH, "avg", i)

    def mirror_models(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

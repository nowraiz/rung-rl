import os

import torch
import random
import math
import torch.optim as optim
import torch.nn.functional as F
from .dqn_network import DQNNetwork
from .replay_memory import ReplayMemory, Transition
from ...obs import Observation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.02
EPS_DECAY = 10000
TARGET_UPDATE = 500
NUM_ACTIONS = 13
INPUTS = 1195
LEARNING_STARTS = 1000
MODEL_PATH = os.getcwd() + "/"


LR = 1e-4


class DQNAgent:
    def __init__(self, player_id, train=True):
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE
        self.num_actions = NUM_ACTIONS
        self.steps_done = 0
        self.policy_net = DQNNetwork(INPUTS, NUM_ACTIONS).cuda()
        self.target_net = DQNNetwork(INPUTS, NUM_ACTIONS).cuda().eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(10000)
        self.id = player_id
        self.eval = train
        self.cards_seen = [None for _ in range(52)]
        self.last_action = None
        self.last_reward = 0
        self.last_state = None
        self.train = train
        self.game_reward = 0
        self.last_game_reward = 0
        self.cards_seen_index = 0
        self.load_model()

    def select_action(self, state, action_mask):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                out = self.policy_net(state)
                mask = torch.tensor([[0 if m == 1 else float("-inf") for m in action_mask]], device=device)
                out = out + mask

                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return out.max(1)[1].view(1, 1)
        else:
            choice = [i for i, _ in enumerate(action_mask) if action_mask[i]]
            return torch.tensor([[random.choice(choice)]], device=device, dtype=torch.long)

    def reward(self, r, done=False):
        self.last_reward = torch.tensor([[r]], dtype=torch.float).to(device)
        self.game_reward += r

    def get_move(self, cards, hand, stack, rung, num_hand, dominating, last_hand, highest, action_mask):
        state = self.get_obs(cards, hand, stack, rung, num_hand, dominating, last_hand, highest)
        if self.last_state is not None:
            self.memory.push(self.last_state, self.last_action, state, self.last_reward)
        self.last_state = state
        self.last_action = self.select_action(self.last_state, action_mask)
        return self.last_action

    def get_obs(self, cards, hand, stack, rung, num_hand, dominating, last_hand, highest):
        obs = Observation(cards, hand, stack, rung, num_hand, dominating, last_hand, highest, self.id, self.cards_seen)
        return torch.tensor([obs.get()], dtype=torch.float).to(device)

    def save_obs(self, hand):
        for card in hand:
            if not card:
                break
            self.save_card(card)

    def save_card(self, card):
        self.cards_seen[self.cards_seen_index] = card
        self.cards_seen_index += 1
        # idx = (card.suit.value - 1) * 13 + card.face.value - 2
        # self.cards_seen[idx] = 1

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return 0
        if not self.train:
            return 0
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward)))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        # print(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # if self.steps_done % 1300 == 0:
        print(loss.item(), end="\t")
        return loss.item()

    def end(self):
        self.memory.push(self.last_state, self.last_action, None, self.last_reward)
        self.last_game_reward = self.game_reward
        self.game_reward = 0
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.cards_seen = [None for _ in range(52)]
        self.cards_seen_index = 0
        # do nothing at the end of the game
        pass

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path())

    def load_model(self):
        try:
            state_dict = torch.load(self.model_path())
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
        except FileNotFoundError:
            print("File not found. Creating a new network...")

    def model_path(self):
        return "{}/model{}".format(MODEL_PATH, self.id)

    def mirror_models(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_rewards(self):
        return self.last_game_reward

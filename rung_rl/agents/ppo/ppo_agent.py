import os
from pickle import TRUE
from rung_rl.obs import Observation
from rung_rl.agents.ppo.ppo_algo import PPO
from typing import List
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from .ppo_network import PPONetwork

# from .rung_network import RungNetwork
# from .replay_memory import ReplayMemory, Transition, ActionMemory, StateAction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# gpu = torch.device("cuda")

# BATCH_SIZE = 64
GAMMA = 0.999
HIDDEN_SIZE = 256
ACTOR_SIZE = 256
CRITIC_SIZE = 256
# EPS_START = 0.3
# EPS_END = 0.05
# EPS_DECAY = 1000000
# TARGET_UPDATE = 1000
# MIN_BUFFER_SIZE = 1000
# RUNG_BATCH_SIZE = 64
GAE = False
GAE_LAMBDA = 0.95
NUM_ACTIONS = 13 + 4
INPUTS = 1486
LEARNING_STARTS = 1000
MODEL_PATH = os.getcwd() + "/models/ppo"
LR = 5e-5

class PPOAgent:
    """
    This class implements an agent that follows the PPO algorithm. It can
    be used to generate arbitrary number of meta-players to play the game
    """

    def __init__(self, train=True) -> None:
        self.GAMMA = GAMMA
        self.HIDDEN_SIZE = HIDDEN_SIZE
        self.ACTOR_SIZE = ACTOR_SIZE
        self.CRITIC_SIZE = CRITIC_SIZE
        self.name = "ppo"
        self.actor = PPONetwork(INPUTS, ACTOR_SIZE, NUM_ACTIONS).to(device)
        self.critic = PPONetwork(INPUTS, CRITIC_SIZE, 1).to(device)
        # self.actor_critic.share_memory()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR)
        self.critic_optmizer = optim.Adam(self.critic.parameters(), lr=LR)
        self.ppo = PPO(self.actor, self.critic, self.actor_optimizer, self.critic_optmizer)
        self.players: List[PPOPlayer] = []  # player that are currently playing the game
        self.state_batch = []
        self.action_batch = []
        self.reward_batch = []
        self.log_probs_batch = []
        self.batch_size = 0
        self.train = train
        self.load_model()

    def get_player(self, train=True):
        """
        Returns a ppo player that can be used to play games using
        ppo agent's brain
            train: if the player will be used for training or not
        """
        player = PPOPlayer(self.actor, self.critic, train)
        if train:  # if the player is to be used in a training game
            self.players.append(player)  # add it to the pool
        return player

    def gather_experience(self):
        """
        Gathers experience replay in the buffers from all the players
        in the pool to optimize the agent on
        """

        for player in self.players:
            self.batch_size += len(player.states)
        # we now have batch size
        self.state_batch = [None for _ in range(self.batch_size)]
        i = 0
        for player in self.players:
            length = len(player.states)
            self.state_batch[i:i+length] = player.states
            self.action_batch[i:i+length] = player.actions
            self.log_probs_batch[i:i+length] = player.log_probs
            self.reward_batch[i:i+length] = player.returns
            i += length
        assert i == self.batch_size

    def clear_experience(self):
        """
        Clears the experience buffers after we are done training.
        Instantly deletes all the players
        """
        self.state_batch = []
        self.action_batch = []
        self.log_probs_batch = []
        self.reward_batch = []
        self.batch_size = 0
        self.players = []  # TODO: Make sure no one can access the existing players

    def clear_players(self):
        """
        Clears all the players from the queue
        """
        self.players = []

    def load_params(self, actor, critic):
        """
        Loads the given params into the actor_critic model directly. Used in parallel environments
        to create arbitrary number of agents with the same parameters
        """
        self.actor.load_state_dict(actor)
        self.critic.load_state_dict(critic)


    def optimize_model_directly(self, state_batch, action_batch, log_probs_batch, reward_batch, batch_size):
        action, value, entropy = self.ppo.update_ppo(state_batch,
                                                        action_batch,
                                                        log_probs_batch,
                                                        reward_batch,
                                                        batch_size)

        print("Action: {}, Value: {}, Entropy: {}".format(action, value, entropy))

    def optimize_model(self):
        if not self.train or self.batch_size < 1:
            return

        action, value, entropy = self.ppo.update_ppo(
            self.state_batch,
            self.action_batch,
            self.log_probs_batch,
            self.reward_batch,
            self.batch_size)

        print("Action: {}, Value: {}, Entropy: {}".format(action, value, entropy))

        self.clear_experience()

    def save_model(self, i="final"):
        torch.save(self.actor.state_dict(), self.model_path(f'{self.name}_actor', i))
        torch.save(self.critic.state_dict(), self.model_path(f'{self.name}_critic', i))
        # torch.save(self.critic.state_dict(), self.model_path("critic"))
        # torch.save(self.average_policy.state_dict(), self.average_model_path(i))

    def load_model(self, i="final"):
        try:
            state_dict = torch.load(self.model_path(f'{self.name}_actor'))
            # self.policy_net.load_state_dict(state_dict)
            self.actor.load_state_dict(state_dict)
            state_dict = torch.load(self.model_path(f'{self.name}_critic'))
            self.critic.load_state_dict(state_dict)
            # self.critic.load_state_dict(state_dict)
            # state_dict = torch.load(self.average_model_path(i))
            # self.average_policy.load_state_dict(state_dict)
        except FileNotFoundError:
            print("File not found. Creating a new network...")

    def model_path(self, model_name, i="final"):
        return "{}/model_{}_{}".format(MODEL_PATH, model_name, i)


class PPOPlayer:
    def __init__(self, actor, critic, train=True):
        self.steps = 0  # the total steps taken by the agent
        self.rewards = []  # rewards acheived at each step
        self.log_probs = []  # log probs of action taken at each step
        self.entropies = []  # entropy of each distribution produced
        self.actions = []  # the actions taken at each step
        self.states = []  # the states (does not really matter)
        self.values = []  # the values predicted by the critic
        self.returns = []  # the returns of each step (after calculation)
        self.dones = []

        self.actor = actor
        self.critic = critic
        self.total_reward = 0
        self.train = train
        self.wins = 0
        self.steps = 0
        self.eval = False
        self.use_gae = GAE
        self.gae_lambda = GAE_LAMBDA

    def get_rung(self, state, player):
        """
        Function responsible for returning the trump suit (i.e. rung) using
        the rung network
        """
        return torch.tensor([random.randint(0, 3)])  # return a random rung for now
        # state = self.get_rung_obs(state)
        # self.rung_state[player] = state
        # self.rung_selected[player] = self.select_rung(self.rung_state[player])
        # return self.rung_selected[player]

    def select_action(self, state, action_mask):
        raw_probs, value = None, None
        with torch.no_grad():
            raw_probs = self.actor(state)
            value = self.critic(state)
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

        action = dist.sample()
        # print(sm(raw_probs))
        # print(action_mask)
        # print(log_probs)
        # return sm(probs).max(1)[1], dist.log_prob(action), value
        return action, dist.log_prob(action), value

    def reward(self, r, player, done=False):
        # self.last_rewards[player] = torch.tensor([[r]], dtype=torch.float).to(device)
        self.total_reward += r
        if self.train:
            self.rewards.append(torch.tensor([r], dtype=torch.float, device=device))
            if done:
                self.values.append(0)
                self.dones.append(0)
            else:
                self.dones.append(1)

    def get_move(self, state):
        # player = state.player_id
        action_mask = state.get_action_mask()
        state = self.get_obs(state)
        # value = self.get_value(state)
        action, log_prob, value = self.select_action(state, action_mask)
        if self.train:
            self.states.append(state)
            self.log_probs.append(log_prob)
            self.actions.append(action)
            self.values.append(value)
        self.steps += 1

        return action

    def create_action_mask_tensor(self, mask):
        return torch.tensor([[0 if m else -1e8 for m in mask]], dtype=torch.float, device=device)

    def get_obs(self, state):
        obs = state.get_obs()
        return torch.tensor([obs.get()], dtype=torch.float, device=device)

    def calculate_returns(self):
        self.returns = [None for _ in range(len(self.rewards))]
        if self.use_gae:
            # generalized advantage estimation
            gae = 0
            for i in range(len(self.rewards) - 1, -1, -1):
                delta = self.rewards[i] + GAMMA * self.values[i + 1] - self.values[i]
                gae = delta + GAMMA * GAE_LAMBDA * gae
                self.returns[i] = gae + self.values[i]
        else:
            # if len(self.rewards[player]) == 1:
            # print(self.rewards[player])
            returns = 0
            for i in range(len(self.rewards) - 1, -1, -1):
                returns = self.rewards[i] + GAMMA * returns
                self.returns[i] = returns

    # def append_transition_batch(self):
    #     """
    #     After a game is finished, poll the individual player transitions to 
    #     create a single transition batch for training in the future
    #     """
    #     if not self.train:
    #         return
    #     for player in range(4):
    #         self.calculate_returns(player) # calculate the returns of the player 
    #         self.all_states.extend(self.states[player])
    #         self.all_actions.extend(self.actions[player])
    #         self.all_log_probs.extend(self.log_probs[player])
    #         self.all_rewards.extend(self.rewards[player])


    def clear(self):
        """
        Clears the trajectory of the player to be used again (not recommended).
        Ideally, a new PPOPlayer should be created for every new training game
        """
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.rewards = []

    def end(self, win, player):
        self.wins += win
        if self.train:
            self.calculate_returns()
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
        wins = self.wins
        reward = self.total_reward
        self.wins = 0
        self.total_reward = 0
        return wins, reward

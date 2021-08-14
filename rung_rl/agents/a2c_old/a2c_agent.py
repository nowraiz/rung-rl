import os

import torch

import rung_rl.utils as utils
from rung_rl.obs import Observation
from .a2c import A2C_ACKTR
from .model import Policy
from .storage import RolloutStorage

args = {}
args["use_gae"] = True
args["cuda"] = False
args["clip_param"] = 0.1
args["ppo_epoch"] = 3
args["num_mini_batch"] = 1
args["value_loss_coef"] = 0.5
args["entropy_coef"] = 0.01
args["gamma"] = 0.995
args["lr"] = 25e-5
args["eps"] = 1e-5
args["max_grad_norm"] = 0.5
args["gae_lambda"] = 0.95
args["num_steps"] = 91
args["num_processes"] = 1
args["alpha"] = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
OBSERVATION_SPACE = 1199
ACTIONS = 13
OBSERVATION_SPACE_SHAPE = torch.Size([OBSERVATION_SPACE])
MODEL_PATH = os.getcwd() + "/models"
MODEL_NAME = MODEL_PATH + "/vfinal.pt"


# actor_critic = None
# try:
#     print("Loading the network from file...")
#     actor_critic = torch.load(MODEL_NAME)
# except FileNotFoundError:
#     print("File not found. Creating a new network...")
#
# actor_critic.to(device)


def save_policy(i, actor_critic, index):
    torch.save(actor_critic, MODEL_PATH + "/v" + str(i) + ".pt")


def update_parameters(i, n, policy):
    utils.update_linear_schedule(policy.optimizer, i, n, args["lr"])
    # utils.update_epsilon(policy.optimizer, i, n, args["eps"])


class A3CAgent:
    def __init__(self, id, eval=False):
        self.actor_critic = Policy(OBSERVATION_SPACE_SHAPE, ACTIONS, base_kwargs={'recurrent': False}).to(device)

        self.policy = policy = A2C_ACKTR(
            self.actor_critic,
            args["value_loss_coef"],
            args["entropy_coef"],
            lr=args["lr"],
            eps=args["eps"],
            alpha=args["alpha"],
            max_grad_norm=args["max_grad_norm"],
            acktr=False)

        self.rollouts = RolloutStorage(args["num_steps"], 1, OBSERVATION_SPACE_SHAPE, ACTIONS,
                                       self.actor_critic.recurrent_hidden_state_size)
        self.rollouts.to(device)
        self.cards_seen = [None for _ in range(52)]
        self.step = 0
        self.rewards = 0
        self.invalid_moves = 0
        self.eval = eval
        self.id = id
        self.fresh = True
        self.card_idx = 0
        self.last_game_reward = 0
        self.obs = None
        self.value = self.action = self.action_log_prob = self.recurrent_hidden_states = None
        self.mask = self.bad_mask = self.reward_tensor = None
        self.wins = 0
        self.last_game_wins = 0
        self.load_model()

    def get_move(self, cards, hand, stack, rung, num_hand, dominating, last_hand, highest, action_mask, last_dominant):
        self.obs = self.get_obs(cards, hand, stack, rung, num_hand, dominating, last_hand, highest, last_dominant)
        if self.step == 0 and self.fresh:
            self.fresh = False
            self.rollouts.obs[0].copy_(self.obs)
            self.rollouts.to(device)
        else:
            self.rollouts.insert(self.obs, self.recurrent_hidden_states, self.action, self.action_log_prob, self.value,
                                 self.reward_tensor, self.mask, self.bad_mask)
            self.step += 1
        with torch.no_grad():
            self.value, self.action, self.action_log_prob, self.recurrent_hidden_states = self.actor_critic.act(
                self.rollouts.obs[self.step], self.rollouts.recurrent_hidden_states[self.step],
                self.rollouts.masks[self.step], action_mask)
        # add this card to the played card
        # self.save_card(cards[self.action])
        return self.action

    def reward(self, r, done=False):
        self.rewards += r
        self.mask = torch.FloatTensor([[0.0 if done else 1.0]]).to(device)
        self.bad_mask = torch.FloatTensor([[1.0]]).to(device)
        self.reward_tensor = torch.FloatTensor([r]).to(device)

    def get_obs(self, cards, hand, stack, rung, num_hand, dominating, last_hand, highest, last_dominant):
        obs = Observation(cards, hand, stack, rung, num_hand, dominating, last_hand, highest, last_dominant,
                          self.id, self.cards_seen)
        return torch.Tensor(obs.get()).to(device)

    def save_obs(self, hand):
        for card in hand:
            if not card:
                break
            self.save_card(card)

    def save_card(self, card):
        # print(self.card_idx)
        self.cards_seen[self.card_idx] = card
        self.card_idx += 1
        # idx = (card.suit.value - 1) * 13 + card.face.value - 2
        # self.cards_seen[idx] = 1

    def train(self):
        if self.eval:  # do not train if in evaluation mode
            self.end_steps()
            return
        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                self.rollouts.masks[-1]).detach()
        self.rollouts.compute_returns(next_value, args["use_gae"], args["gamma"],
                                      args["gae_lambda"], False)

        value_loss, action_loss, dist_entropy = self.policy.update(self.rollouts)
        print(utils.format_list([value_loss, action_loss, dist_entropy]), end="\t")
        self.end_steps()

    def end(self, win):
        self.wins += int(win)
        self.cards_seen = [None for _ in range(52)]
        self.card_idx = 0

    def end_steps(self):
        self.last_game_wins = self.wins
        self.wins = 0
        self.last_game_reward = self.rewards
        self.rewards = 0
        self.step = 0
        self.rollouts.after_update()

    def get_rewards(self):
        return self.last_game_reward

    def model_path(self):
        return "{}/model{}".format(MODEL_PATH, self.id)

    def save_model(self):
        torch.save(self.actor_critic.state_dict(), self.model_path())

    def load_model(self):
        try:
            state_dict = torch.load(self.model_path())
            self.actor_critic.load_state_dict(state_dict)
            print("Loading the network from file...")
        except FileNotFoundError:
            print("Creating a new network...")

    def get_wins(self):
        return self.last_game_wins

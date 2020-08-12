from rung_rl.agents.a2c.a2c_agent import A2CAgent
from rung_rl.agents.dqn.dqn_agent import DQNAgent
from rung_rl.agents.human_agent import HumanAgent
from rung_rl.agents.random_agent import RandomAgent
from rung_rl.rung import Game
import torch
import torch.multiprocessing as mp

import numpy as np
from multiprocessing import Process

CONCURRENT_GAMES = 128


def run_game(players):
    game = Game(players)
    game.initialize()
    game.play_game()


# def train(num_games, num_processes):
#     model_version = 0
#     print("Starting training...")
#     for i in range(num_games // num_processes):
#         # pool = Pool(processes=4)
#         if i % (num_games // num_processes / 20) == 0:
#             save_policy(model_version)
#             model_version += 1
#         processes = []
#         for j in range(num_processes):
#             p = Process(target=run_game())
#             p.start()
#             processes.append(p)
#         for p in processes:
#             p.join()  # wait for each game to finish
#         print(i * num_processes)
#     save_policy("final")
#     evaluate(1000)
#
#
# def train_sequential(num_games):
#     agents = []
#     # ai_vs_shit = [ai_vs_ai[0], RandomAgent(1), ai_vs_ai[2], RandomAgent(4)]
#     model_version = 0
#     for i in range(num_games // CONCURRENT_GAMES):
#         agents = []
#         for j in range(CONCURRENT_GAMES):
#             ai_vs_shit = [PPOAgent(0), RandomAgent(1), PPOAgent(2), RandomAgent(3)]
#             ai_vs_ai = [PPOAgent(0), PPOAgent(1), PPOAgent(2), PPOAgent(3)]
#             if i % (num_games / 20) == 0:
#                 save_policy(model_version)
#                 model_version += 1
#             print(i * CONCURRENT_GAMES + j)
#             if i % 5 == 0:
#                 run_game(ai_vs_shit)
#                 agents += ai_vs_shit
#             else:
#                 run_game(ai_vs_ai)
#                 agents += ai_vs_ai
#             update_parameters((i * CONCURRENT_GAMES) + j, num_games)
#         for agent in agents:
#             agent.train()
#
#     evaluate(1000)
#     save_policy("final")


def train_a2c(num_games, debug=False):
    print("Starting training")
    # players = [A2CAgent(0), A2CAgent(1), A2CAgent(2), DQNAgent(3)]
    # players4 = [RandomAgent(0), RandomAgent(1), RandomAgent(2), RandomAgent(4)]
    players = [A2CAgent(0), A2CAgent(1), A2CAgent(2), A2CAgent(3)]
    players2 = [players[0], RandomAgent(1), players[2], RandomAgent(3)]
    players3 = [RandomAgent(0), players[1], RandomAgent(2), players[3]]
    wins = 0
    reward = 0
    for i in range(num_games):
        game = Game(players, debug, debug)
        game.initialize()
        game.play_game()
        rewards = [players[0].get_rewards(), players[1].get_rewards()]
        wins += int(rewards[0] == max(rewards))
        reward += rewards[0]
        if i % 250 == 0:
            print("Last 250 games win rate: {}. Rewards {}. Total games {}"
                  .format(wins / 250, reward / 250, i))
            wins = 0
            reward = 0
        for player in players:
            if type(player) == A2CAgent:
                player.train()
        # print()
        if i % 5000 == 0:
            for player in players:
                if type(player) == A2CAgent:
                    player.save_model()
        if i % 10000 == 0 and i != 0:
            random_wins_team_0 = strategy_collapse(players2, 0)
            random_wins_team_1 = strategy_collapse(players3, 1)
            print("Evaluation of 200 games with random: Team_radiant_0_2 {}, Team_dire_1_3 {}"
                  .format(random_wins_team_0/200, random_wins_team_1/200))
    for player in players:
        if type(player) == A2CAgent:
            player.save_model()
    print("Starting evaluation")
    evaluate(1000, players)


def strategy_collapse(players, idx):
    wins = 0
    for i in range(200):
        game = Game(players)
        game.initialize()
        game.play_game()
        for player in players:
            if type(player) == DQNAgent:
                player.optimize_model()
        rewards = [players[0].get_rewards(), players[1].get_rewards()]
        wins += int(rewards[idx] == max(rewards))
        print()
    return wins


def evaluate(num_games, players, debug=False):
    print("Starting evaluation...")
    wins = 0
    r = 0
    for _ in range(num_games):
        # players = [PPOAgent(0, True), RandomAgent(1), PPOAgent(2, True), RandomAgent(3)]
        game = Game(players, debug, debug)
        game.initialize()
        game.play_game()
        rewards = [players[0].get_rewards(), players[1].get_rewards()]
        win = int(rewards[0] == max(rewards))
        wins += win
        # r += rewards[0]
        print(rewards[0])

    print(wins, wins / num_games, r, r / num_games)

def test():
    players = [A2CAgent(0), RandomAgent(1), RandomAgent(2), RandomAgent(3)]
    game = Game(players)
    game.initialize()
    game.play_game()
    players[0].train()

def play_game():
    players = [RandomAgent(0), HumanAgent(1), RandomAgent(2), RandomAgent(3)]
    game = Game(players, True, True)
    game.initialize()
    game.play_game()


if __name__ == "__main__":
    train_a2c(1000000-20000)
    # train_a2c(1)
    # players = [A2CAgent(0, False), RandomAgent(1), A2CAgent(2, False), RandomAgent(3)]
    # players = [RandomAgent(0), A2CAgent(0, True), RandomAgent(1), A2CAgent(1, True)]
    # evaluate(1000, players)
    # players = [RandomAgent(0), DQNAgent(1, False), RandomAgent(2), DQNAgent(3, False)]
    # players = [DQNAgent(0, False), RandomAgent(1), DQNAgent(2, False), RandomAgent(3)]
    # players = [RandomAgent(0), RandomAgent(1), RandomAgent(2), RandomAgent(3)]
    # evaluate(1000, players, False)

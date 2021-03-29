from rung_rl.agents.a2c.a2c_agent import A2CAgent
from rung_rl.agents.human_agent import HumanAgent
from rung_rl.agents.random_agent import RandomAgent
from rung_rl.rung import Game
import rung_rl.plotter as plt
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
    players = [A2CAgent(), A2CAgent(), A2CAgent(), A2CAgent()]
    win_rate_radiant = []
    win_rate_dire = []
    games = []
    for i in range(num_games):
        game = Game(players, False, False)
        game.initialize()
        game.play_game()

        if i % 6 == 0:
            for player in players:
                player.optimize_model()
            print()
        if i % 100 == 0:
            print("Total Games: {}".format(i))

        if i % 5000 == 0:
            players[0].save_model("final")
            print("Steps done: {}".format(players[0].steps))

        if i % 5000 == 0 and i != 0:
            temp_players_radiant = [players[0], RandomAgent(1), players[2], RandomAgent(3)]
            temp_players_dire = [RandomAgent(0), players[1], RandomAgent(2), players[3]]
            # for player in players:
            #     player.eval = True
            win_rate_r = evaluate(100, temp_players_radiant, 0)
            win_rate_d = evaluate(100, temp_players_dire, 1)
            games.append(i)
            win_rate_radiant.append(win_rate_r)
            win_rate_dire.append(win_rate_d)

            plt.plot(games, win_rate_radiant, win_rate_dire)
            plt.savefig()
            # for player in players:
            #     player.eval = False

    plt.savefig()
    players[0].save_model("final")
    print("Steps done: {}".format(players[0].steps))
    

def test():
    agent = DQNAgent()
    agent.load_model()
    agent.eval = True
    players = [agent, agent, agent, agent]
    game = Game(players)
    game.initialize()
    game.play_game()


def evaluate(num_games, players, idx=0, debug=False):
    print("Starting evaluation...")
    players[idx].reset()
    for i in range(num_games):
        game = Game(players, debug, debug)
        game.initialize()
        game.play_game()

    wins = players[idx].reset()
    print(wins, wins / num_games)
    return wins / num_games * 100


def play_game():
    players = [RandomAgent(0), HumanAgent(1), RandomAgent(2), RandomAgent(3)]
    game = Game(players, True, True)
    game.initialize()
    game.play_game()


if __name__ == "__main__":
    train_a2c(1000000)
    # test()
    agent = A2CAgent()
    agent.load_model("final")
    # agent.deterministic = True
    players = [agent, RandomAgent(1), agent, RandomAgent(2)]
    # # players = [DQNAgent(0, False), RandomAgent(1), DQNAgent(2, False), RandomAgent(3)]
    # # players = [RandomAgent(0), agent, RandomAgent(2), agent]
    evaluate(100, players, 0, False)

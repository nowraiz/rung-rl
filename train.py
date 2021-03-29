from rung_rl.agents.dqn.dqn_agent import DQNAgent
from rung_rl.agents.human_agent import HumanAgent
from rung_rl.agents.random_agent import RandomAgent
from rung_rl.rung import Game
import rung_rl.plotter as plt
import torch
# import multiprocessing as mp

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
        # pool = Pool(processes=4)
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


def train_dqn(num_games, debug=False):
    win_rate_radiant = []
    win_rate_dire = []
    games = []

    plt.plot(games, win_rate_radiant, win_rate_dire)
    print("Starting training")
    agent = DQNAgent()
    players = [agent, agent, agent, agent]
    wins = 0
    reward = 0
    i = 0
    # players[0].save_model("weak")
    while 1:
        game = Game(players, debug, debug)
        game.initialize()
        game.play_game()
        agent.optimize_model()
        agent.optimize_rung_network()
        print()
        # rewards = [players[0].get_rewards(), players[1].get_rewards()]
        # wins += int(rewards[0] == max(rewards))
        # reward += rewards[0]
        if i % 100 == 0:
        #     print("Last 100 games win rate: {}. Rewards {}. Total games {}]"
        #           .format(wins / 100, reward / 100, i))
            print("Total Games: {}".format(i))
        
            # win_rate_radiant.append(random_wins_team_0*100)
            # win_rate_dire.append(random_wins_team_1*100)
            # games.append(i)
            # plt.plot(games, win_rate_radiant, win_rate_dire)
            
        if i % 5000 == 0 and i != 0:
            players[0].save_model("final")
            for player in players:
                player.eval = True
            players2 = [players[0], RandomAgent(1), players[2], RandomAgent(3)]
            players3 = [RandomAgent(0), players[1], RandomAgent(2), players[3]]
            win_rate_r = evaluate(100, players2, 0)
            win_rate_d = evaluate(100, players3, 1)
            win_rate_radiant.append(win_rate_r)
            win_rate_dire.append(win_rate_d)
            games.append(i)
            plt.plot(games, win_rate_radiant, win_rate_dire)
            plt.savefig()
            for player in players:
                player.eval = False

        if i % 500 == 0:
            players[0].mirror_models()
            # for player in players:
            #     if type(player) == DQNAgent:
            #         player.mirror_models()
        # if i % 8000 == 0 and i != 0:
        #     weak_agent = DQNAgent()
        #     weak_agent.load_model("weak")  # load the previous saved agent
        #     players2 = [agent, weak_agent, agent, weak_agent]
        #     players3 = [weak_agent, agent, weak_agent, agent]
        #     weak_wins_team_0 = (strategy_collapse(players2, 0) / 1000) / 2
        #     for player in players:
        #         if type(player) == DQNAgent:
        #             player.mirror_models()
        #     weak_wins_team_1 = (strategy_collapse(players3, 1) / 1000) / 2

        #     for player in players:
        #         if type(player) == DQNAgent:
        #             player.mirror_models()

        #     players[0].save_model("weak")
        #     players[0].save_model("final")

        #     i += 2000
        i += 1

    for player in players:
        if type(player) == DQNAgent:
            player.save_model("final")
    # print("Starting evaluation")
    # agent.deterministic = True
    # evaluate(1000, players2)


def strategy_collapse(players, idx):
    wins = 0
    players[idx].reset()
    for i in range(1000):
        game = Game(players)
        game.initialize()
        game.play_game()
        players[idx].optimize_model()
        # rewards = [players[0].get_rewards(), players[1].get_rewards()]
        # wins += int(rewards[0] == max(rewards))
        # print()
    return players[idx].reset()


def evaluate(num_games, players, idx=0, debug=False):
    print("Starting evaluation...")
    players[idx].reset()
    winner = None
    for i in range(num_games):
        game = Game(players, debug, debug)
        game.initialize(winner)
        winner = game.play_game()
        # rewards = [players[0].get_rewards(), players[1].get_rewards()]
        # wins += int(rewards[0] == max(rewards))
        # print()
    wins = players[idx].reset()
    print(wins/2 , (wins / 2) / num_games * 100)
    return wins / 2


def play_game(players):
    # players = [HumanAgent, HumanAgent(1), RandomAgent(2), RandomAgent(3)]
    game = Game(players, True, True)
    game.initialize(0)
    game.play_game()


if __name__ == "__main__":
    # train_dqn(1)
    # agent = DQNAgent(False)
    # # agent2 = DQNAgent(False)
    # agent.load_model("final")
    # # agent2.load_model()
    # agent.eval = True
    # # players = [HumanAgent(0), agent, agent, agent]
    # players = [RandomAgent(0), agent, RandomAgent(2), agent]
    # # play_game(players)
    # # # players = [RandomAgent(0), agent, RandomAgent(2), agent]
    # # # # # # players = [DQNAgent(0, False), RandomAgent(1), DQNAgent(2, False), RandomAgent(3)]
    # # # # # # players = [RandomAgent(0), agent, RandomAgent(2), agent]
    # evaluate(1, players, 1, False)

from rung_rl.agents.abstract_agent import Agent
from rung_rl.agents.dqn.dqn_agent import DQNAgent
from rung_rl.agents.human_agent import HumanAgent
from rung_rl.agents.random_agent import RandomAgent
from rung_rl.agents.retarded_agent import RetardedAgent
from rung_rl.agents.oracle.oracle import Oracle
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

    weak_agent = DQNAgent(False, True)
    weak_agent.eval = True
    # plt.plot(games, win_rate_radiant, win_rate_dire)
    print("Starting training")
    agent = DQNAgent(True, True) # to indicate that we want to train the agent
    agent.save_model("weak")
    players = [agent, agent, agent, agent]
    # secondary_players = [agent, weak_agent, agent, weak_agent]
    wins = 0
    reward = 0
    rewards = []
    plt.plot(games, win_rate_radiant, win_rate_dire)
    i = 0
    games_i = 0
    # players[0].save_model("weak")
    while 1:
        game = Game(players, debug, debug)
        game.initialize()
        game.play_game()
        agent.optimize_model()
        # agent.optimize_rung_network()
                # print()
        # rewards = [players[0].get_rewards(), players[1].get_rewards()]
        # wins += int(rewards[0] == max(rewards))
        # reward += rewards[0]
        if i % 250 == 0:
            agent.mirror_models()

        # if i % 800 == 0 and i != 0:
        #     # strategy_collapse
        #     weak_agent.load_model("weak")
        #     for _ in range(200):
        #         secondary_game = Game(secondary_players, debug, debug)
        #         secondary_game.initialize()
        #         secondary_game.play_game()
        #         for _ in range(1):
        #             agent.optimize_model()
        #             agent.optimize_rung_network()
        #             print()
        #         # i += 1
        #     agent.save_model("weak")
        if i % 300 == 0:
        #     print("Last 100 games win rate: {}. Rewards {}. Total games {}]"
        #           .format(wins / 100, reward / 100, i))
            print("Total Games: {}".format(i))
        
            # win_rate_radiant.append(random_wins_team_0*100)
            # win_rate_dire.append(random_wins_team_1*100)
            # games.append(i)
            # plt.plot(games, win_rate_radiant, win_rate_dire)
            
        if i % 10000 == 0 and i != 0:
            players[0].save_model("final")
            agent.eval = True
            agent.train = False
            # for player in players:
            #     player.eval = True
            #     player.train = False
            weak_agent.load_model("weak")
            players3 = [agent, weak_agent, agent, weak_agent]
            win_rate_r, _ = evaluate(500, players3, 0)
            players2 = [agent, RandomAgent(), agent, RandomAgent()]
            win_rate_d, _ = evaluate(500, players2, 0)
            # rewards.append(avg_reward)
            win_rate_radiant.append(win_rate_r/100)
            win_rate_dire.append(win_rate_d/100)
            # win_rate_d = evaluate(100, players3, 1)
            # win_rate_radiant.append(win_rate_r)
            # win_rate_dire.append(win_rate_d)
            games.append(games_i)
            plt.plot(games, win_rate_radiant,win_rate_dire)
            plt.savefig()
            agent.eval = False
            agent.train = True

            if win_rate_r < 50:
                # if the previous agent beats you, train against that
                strategy_collapse(players3, agent)
                games_i += 2500

            agent.save_model("weak")

            # for player in players:
            #     player.eval = False
            #     player.train = True

        
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
        games_i += 1

    # print("Starting evaluation")
    # agent.deterministic = True
    # evaluate(1000, players2)


def strategy_collapse(players, agent):
    wins = 0
    # agent.reset()
    for i in range(2500):
        game = Game(players)
        game.initialize()
        game.play_game()
        agent.optimize_model()

        if i % 250 == 0:
            agent.mirror_models()
        # rewards = [players[0].get_rewards(), players[1].get_rewards()]
        # wins += int(rewards[0] == max(rewards))
        # print()
    # return agent.reset()

def evaluate(num_games, players, idx=0, debug=False):
    print("Starting evaluation...")
    wins = 0
    toss = None
    for i in range(num_games):
        game = Game(players, debug, debug)
        game.initialize(toss)
        winners = game.play_game()
        if idx in winners:
            wins += 1
        toss = winners[0]

    avg_reward = 0
    print(wins, wins / num_games, avg_reward)
    return wins / num_games * 100, avg_reward


def play_game(players):
    # players = [HumanAgent, HumanAgent(1), RandomAgent(2), RandomAgent(3)]
    game = Game(players, True, True)
    game.initialize(0)
    game.play_game()


if __name__ == "__main__":
    # pass
    train_dqn(1)
    oracle = DQNAgent(False, True, True)
    dqn_agent = DQNAgent(False, True, False)
    oracle.load_model("oracle")
    oracle.eval= True
    dqn_agent.load_model("normal")
    # agent = Oracle(False)
    dqn_agent.eval = True
    # agent.eval = True
    players = [oracle, dqn_agent, oracle, dqn_agent]
    # players = [dqn_agent, HumanAgent(1), dqn_agent, dqn_agent]
    evaluate(1, players, 0, True)
    # # # agent2 = DQNAgent(False)
    # agent.load_model("final")
    # # # agent2.load_model()
    # agent.eval = True
    # # players = [HumanAgent(0), agent, agent, agent]
    # players = [RandomAgent(0), agent, RandomAgent(2), agent]
    # # play_game(players)
    # # # # players = [RandomAgent(0), agent, RandomAgent(2), agent]
    # # # # # # # players = [DQNAgent(0, False), RandomAgent(1), DQNAgent(2, False), RandomAgent(3)]
    # # # # # # # players = [RandomAgent(0), agent, RandomAgent(2), agent]
    # evaluate(1000, players, 1, False)

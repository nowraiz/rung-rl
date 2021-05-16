from multiprocessing import process
from rung_rl.agents.a2c.a2c_agent import A2CAgent
from rung_rl.agents.ppo.ppo_agent import PPOAgent, PPOPlayer
from rung_rl.agents.dqn.dqn_agent import DQNAgent
from rung_rl.agents.human_agent import HumanAgent
from rung_rl.agents.random_agent import RandomAgent
from rung_rl.rung import Game
import rung_rl.plotter as plt
import torch
# import torch.multiprocessing as mp
import statistics
import numpy as np
import time
# from multiprocessing import Pool
# import multiprocessing as mp
# import torch.multiprocessing as mp


CONCURRENT_GAMES = 64
PROCESSES = 6

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

def play_game(players):
    game = Game(players)
    game.initialize()
    game.play_game()


def strategy_collapse(agent, weak_agent, num_games):
    print("Strategy Collapse...")
    for i in range(num_games):
        players = [
            agent.get_player(),
            weak_agent.get_player(False),
            agent.get_player(),
            weak_agent.get_player(False)
        ]
        game = Game(players)
        game.initialize()
        game.play_game()

        if (i % CONCURRENT_GAMES) == 0:
            agent.gather_experience()
            agent.optimize_model()
    agent.save_model("weak")

def train_a2c(num_games, debug=False):
    agent = PPOAgent()
    weak_agent = PPOAgent()
    agent.save_model("weak")
    dqn_agent = DQNAgent(True)
    dqn_agent.eval = True
    # players = [agent, agent, agent, agent]
    win_rate_radiant = []
    win_rate_dire = []
    games = []
    rewards = []
    wins = 0
    win_rate = []
    avg_rewards_r = []
    avg_rewards_d = []
    then = time.time()
    for i in range(num_games):
        players = [
            agent.get_player(),
            agent.get_player(),
            agent.get_player(),
            agent.get_player()
        ]
        game = Game(players)
        game.initialize()
        game.play_game()
        # processes = []
        # for rank in range(CONCURRENT_GAMES):
        #     players = [
        #         RandomAgent(0),
        #         RandomAgent(1),
        #         RandomAgent(2),
        #         RandomAgent(3)
        #     ]
        #     p = mp.Process(target=play_game, args=(players,))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
        if (i % CONCURRENT_GAMES) == 0 and i != 0:
            agent.gather_experience()
            agent.optimize_model()
            now = time.time()
            print("Time elapsed for CONCURRENT_GAMES games: {}".format(now-then))
            then = now

        if (i % (CONCURRENT_GAMES * 80) == 0 and i != 0):
            #
            weak_agent.load_model("weak")
            temp_players = [
                weak_agent.get_player(False),
                agent.get_player(False),
                weak_agent.get_player(False),
                agent.get_player(False)
            ]
            win_rate_self, reward_self = evaluate(100, temp_players, 1)
            games.append(i/CONCURRENT_GAMES)
            win_rate_radiant.append(win_rate_self/100)
            avg_rewards_r.append(reward_self)
            plt.plot_reward(games, win_rate_radiant, avg_rewards_r)
            plt.savefig()
            agent.clear_experience()
            # strategy collapse
            strategy_collapse(agent, weak_agent, CONCURRENT_GAMES*20)
        
        if ((i / CONCURRENT_GAMES) % 25) == 0:
            agent.save_model("final")


        if i % (CONCURRENT_GAMES * 4) == 0:
            print("Total Games: {}".format(i))
 
        # if (i / CONCURRENT_GAMES) % 100 == 0 and i != 0:
        #     agent.save_model("final")
        #     # print("Steps done: {}".format(players[0].steps))
        #     print("Total Games: {}".format(i))

        #     temp_players_radiant = [   
        #         agent.get_player(False), 
        #         RandomAgent(1), 
        #         agent.get_player(False), 
        #         RandomAgent(3)]

        #     temp_players_dire = [
        #         dqn_agent, 
        #         agent.get_player(False), 
        #         dqn_agent, 
        #         agent.get_player(False)]
        # #     # for player in players:
        # #     #     player.eval = True
        #     win_rate_r, reward_r = evaluate(100, temp_players_radiant, 0)
        #     win_rate_d, reward_d = evaluate(100, temp_players_dire, 1)
        #     games.append(i/CONCURRENT_GAMES) # actually updates
        #     # win_rate_radiant.append(win_rate_r)
        #     # win_rate_dire.append(win_rate_d)
        #     avg_rewards_r.append(reward_r)
        #     avg_rewards_d.append(reward_d)

        #     plt.plot_reward(games, avg_rewards_r, avg_rewards_d)
        #     plt.savefig()
        #     agent.clear_experience()
            # agent.train = True
        #     # for player in players:
        #     #     player.eval = False

    plt.savefig()
    agent.save_model("final")
    print("Steps done: {}".format(players[0].steps))
    


def experiments():
    games = 10000
    steps = 0
    players = [RandomAgent(0), RandomAgent(1), RandomAgent(2), RandomAgent(3)]
    for i in range(games):
        game = Game(players)
        game.initialize()
        game.play_game()
        steps += players[0].get_steps()
    return steps / games


def evaluate(num_games, players, idx=0, debug=False):
    print("Starting evaluation...")
    # players[idx].reset(idx)
    wins = 0
    toss = None
    for i in range(num_games):
        game = Game(players, debug, debug)
        game.initialize(toss)
        winners = game.play_game()
        if idx in winners:
            wins += 1
        # toss = winners[0]

    # wins, _ = players[idx].reset(idx)
    avg_reward = players[idx].total_reward / num_games
    print(wins, wins / num_games, avg_reward)
    return wins / num_games * 100, avg_reward

def main():
    # mp.set_start_method('spawn')
    train_a2c(10000000)
    # test()
    agent = PPOAgent(False)
    # agent.load_model("final")
    # agent.eval =True
    dqn_agent = DQNAgent()
    dqn_agent.eval = True
    # agent.deterministic = True
    players = [agent.get_player(False), RandomAgent(1), agent.get_player(False), RandomAgent(2)]
    evaluate(1000, players, 0, False)
    # players = [agent.get_player(False), dqn_agent, agent.get_player(False), dqn_agent]
    # print("Vs DQN")
    # evaluate(1000, players, 0, False)
    # # players = [RandomAgent(0), agent, RandomAgent(2), agent]

if __name__ == "__main__":
    main()    

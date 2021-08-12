from random import Random
from rung_rl.agents.ppo.ppo_algo import PPO
from torch.multiprocessing import Pipe, Queue
from rung_rl.agents.a2c.a2c_agent import A2CAgent
from rung_rl.agents.ppo.ppo_agent import PPOAgent, PPOPlayer
from rung_rl.agents.dqn.dqn_agent import DQNAgent
from rung_rl.agents.human_agent import HumanAgent
from rung_rl.agents.random_agent import RandomAgent
from rung_rl.env import RungEnv
from rung_rl.rung import Game
import rung_rl.plotter as plt
import torch
# import torch.multiprocessing as mp
import statistics
import numpy as np
import time
# from multiprocessing import Pool
# import multiprocessing as mp
import torch.multiprocessing as mp

PROCESSES = 4 
CONCURRENT_GAMES = PROCESSES * 8


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
    games_list = []
    win_rate_dire = []
    games = []
    rewards = []
    wins = 0
    win_rate = []
    avg_rewards_r = []
    avg_rewards_d = []
    then = time.time()
    processes = []
    pipes = []
    queues = []
    state_batch = []
    reward_batch = []
    log_prob_batch = []
    action_batch = []
    batch_size = 0
    # pipe is (input, output)
    for i in range(PROCESSES):
        queue = Queue()
        (conn1, conn2) = Pipe()
        p = RungEnv(conn2, queue)
        processes.append(p)
        pipes.append(conn1)
        queues.append(queue)
        p.start()

    # start the games

    # which of the processes are running the games
    running = [True for _ in range(PROCESSES)]
    total_games = 0
    for i in range(num_games):
        # loops
        games = 0
        # send the newest parameters:
        for i, p in enumerate(processes):
            pipe = pipes[i]
            q = queues[i]
            pipe.send("REFRESH")
            q.put(agent.actor.state_dict())
            q.put(agent.critic.state_dict())
            pipe.send("RESET")
            running[i] = True
            games += 1

        while games < CONCURRENT_GAMES:
            for i, p in enumerate(processes):
                pipe = pipes[i]
                if running[i]:
                    if (pipe.poll()):
                        q = queues[i]
                        # game finished
                        msg = pipe.recv()
                        assert msg == "END"
                        state_game_batch = q.get()
                        action_game_batch = q.get()
                        reward_game_batch = q.get()
                        log_prob_game_batch = q.get()
                        state_batch += state_game_batch
                        action_batch += action_game_batch
                        reward_batch += reward_game_batch
                        log_prob_batch += log_prob_game_batch

                        if games < CONCURRENT_GAMES:
                            pipe.send("RESET")  # start a new game
                            games += 1
                        else:
                            running[i] = False

        while any(running):
            for i, p in enumerate(processes):
                pipe = pipes[i]
                if running[i]:
                    if (pipe.poll()):
                        q = queues[i]
                        # game finished
                        msg = pipe.recv()
                        assert msg == "END"
                        state_game_batch = q.get()
                        action_game_batch = q.get()
                        reward_game_batch = q.get()
                        log_prob_game_batch = q.get()
                        state_batch += state_game_batch
                        action_batch += action_game_batch
                        reward_batch += reward_game_batch
                        log_prob_batch += log_prob_game_batch

                        running[i] = False

        total_games += games
        print(f'Total games: {total_games}')
        agent.optimize_model_directly(state_batch, action_batch, log_prob_batch, reward_batch, len(state_batch))

        state_batch = []
        action_batch = []
        log_prob_batch = []
        reward_batch = []

        if (total_games % 5040) == 0:
            players = [agent.get_player(False), RandomAgent(), agent.get_player(False), RandomAgent()]
            win_rate_r, _ = evaluate(100, players, 0, False)

            games_list.append(total_games)
            win_rate_radiant.append(win_rate_r)
            plt.plot(games_list, win_rate_radiant, None)
            agent.clear_experience()
            agent.save_model("final")
        # if (i % (CONCURRENT_GAMES * 80) == 0 and i != 0):
        #     #
        #     weak_agent.load_model("weak")
        #     temp_players = [
        #         weak_agent.get_player(False),
        #         agent.get_player(False),
        #         weak_agent.get_player(False),
        #         agent.get_player(False)
        #     ]
        #     win_rate_self, reward_self = evaluate(100, temp_players, 1)
        #     games.append(i/CONCURRENT_GAMES)
        #     win_rate_radiant.append(win_rate_self/100)
        #     avg_rewards_r.append(reward_self)
        #     plt.plot_reward(games, win_rate_radiant, avg_rewards_r)
        #     plt.savefig()
        #     agent.clear_experience()
        #     # strategy collapse
        #     strategy_collapse(agent, weak_agent, CONCURRENT_GAMES*20)


        # if i % (CONCURRENT_GAMES * 4) == 0:

    # plt.savefig()
    agent.save_model("final")
    # print("Steps done: {}".format(players[0].steps))


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
    # train_a2c(10000000)
    # test()
    agent = PPOAgent(False)
    # agent.load_model("final")
    # agent.eval =True
    # agent.deterministic = True
    players = [agent.get_player(False), RandomAgent(), agent.get_player(False), RandomAgent()]
    evaluate(1000, players, 0, False)
    # players = [agent.get_player(False), dqn_agent, agent.get_player(False), dqn_agent]
    # print("Vs DQN")
    # evaluate(1000, players, 0, False)
    # # players = [RandomAgent(0), agent, RandomAgent(2), agent]


if __name__ == "__main__":
    main()

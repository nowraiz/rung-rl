import torch

import rung_rl.plotter as plt
from rung_rl.agents.a2c.a2c_agent import A2CAgent
from rung_rl.agents.dqn.dqn_agent import DQNAgent
from rung_rl.agents.human_agent import HumanAgent
from rung_rl.agents.random_agent import RandomAgent
from rung_rl.rung import Game

torch.set_num_threads(12)

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
    agent = A2CAgent()
    dqn_agent = DQNAgent(True)
    dqn_agent.eval = True
    players = [agent, agent, agent, agent]
    win_rate_radiant = []
    win_rate_dire = []
    games = []
    rewards = []
    wins = 0
    win_rate = []
    avg_rewards = []
    for i in range(num_games):
        game = Game(players, False, False)
        game.initialize()
        game.play_game()

        if i % 6 == 0:
            agent.optimize_model()
            print()
        # agent.reset(0)
        # temp_players_radiant = [agent, RandomAgent(1), agent, RandomAgent(3)]
        # test_game = Game(temp_players_radiant, False, False)
        # test_game.initialize()
        # test_game.play_game()
        # win, reward = agent.reset(0)
        # wins += win
        # # rewards.append(reward)
        # # games.append(i)
        # rewards.append(reward)
        # # avg_reward = sum(avg_reward) / 100
        # # avg_rewards.append(avg_reward)

        # agent.clear_trajectory()

        if i % 100 == 0:
            print("Total Games: {}".format(i))
        # if i % 250 == 0 and i != 0:
        #     games.append(i)
        #     avg_reward = statistics.mean(rewards)
        #     avg_rewards.append(avg_reward)
        #     rewards = []
        #     win_rate.append(wins / 250)
        #     wins = 0
        #     plt.plot_reward(games, avg_rewards, win_rate)
        #     plt.savefig()

        if i % 5000 == 0:
            players[0].save_model("final")
            print("Steps done: {}".format(players[0].steps))

        if i % 5000 == 0 and i != 0:
            temp_players_radiant = [players[0], RandomAgent(), players[2], RandomAgent()]
            temp_players_dire = [dqn_agent, players[1], dqn_agent, players[3]]
            #     # for player in players:
            #     #     player.eval = True
            win_rate_r = evaluate(100, temp_players_radiant, 0)
            win_rate_d = evaluate(100, temp_players_dire, 1)
            games.append(i)
            win_rate_radiant.append(win_rate_r)
            win_rate_dire.append(win_rate_d)

            plt.plot(games, win_rate_dire, win_rate_radiant)
            plt.savefig()
            agent.clear_trajectory()
        #     # for player in players:
        #     #     player.eval = False

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
    players[idx].reset(idx)
    for i in range(num_games):
        game = Game(players, debug, debug)
        game.initialize()
        game.play_game()

    wins, _ = players[idx].reset(idx)
    print(wins, wins / num_games)
    return wins / num_games * 100


def play_game():
    players = [RandomAgent(0), HumanAgent(1), RandomAgent(2), RandomAgent(3)]
    game = Game(players, True, True)
    game.initialize()
    game.play_game()


if __name__ == "__main__":
    # train_a2c(10000000)
    # test()
    agent = A2CAgent()
    agent.load_model("final")
    agent.eval = True
    # agent.deterministic = True
    players = [agent, RandomAgent(), agent, RandomAgent()]
    evaluate(10000, players, 0, False)
    players = [agent, dqn_agent, agent, dqn_agent]
    print("Vs DQN")
    evaluate(10000, players, 0, False)
    # # players = [RandomAgent(0), agent, RandomAgent(2), agent]

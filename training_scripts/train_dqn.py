import rung_rl.plotter as plt
from rung_rl.agents.dqn.dqn_agent import DQNAgent
from rung_rl.agents.random_agent import RandomAgent
from rung_rl.rung import Game


CONCURRENT_GAMES = 128


def train_dqn(num_games, debug=False):
    win_rate_radiant = []
    win_rate_dire = []
    games = []

    weak_agent = DQNAgent(False, True)
    weak_agent.eval = True
    print("Starting training")
    agent = DQNAgent(True, True)  # to indicate that we want to train the agent
    agent.save_model("weak")
    players = [agent, agent, agent, agent]
    plt.plot(games, win_rate_radiant, win_rate_dire)
    i = 0
    games_i = 0
    while 1:
        game = Game(players, debug, debug)
        game.initialize()
        game.play_game()
        agent.optimize_model()
        if i % 250 == 0:
            agent.mirror_models()

        if i % 300 == 0:
            print("Total Games: {}".format(i))

        if i % 10000 == 0 and i != 0:
            players[0].save_model("final")
            agent.eval = True
            agent.train = False
            weak_agent.load_model("weak")
            players3 = [agent, weak_agent, agent, weak_agent]
            win_rate_r, _ = evaluate(500, players3, 0)
            players2 = [agent, RandomAgent(), agent, RandomAgent()]
            win_rate_d, _ = evaluate(500, players2, 0)
            win_rate_radiant.append(win_rate_r / 100)
            win_rate_dire.append(win_rate_d / 100)
            games.append(games_i)
            plt.plot(games, win_rate_radiant, win_rate_dire)
            plt.savefig()
            agent.eval = False
            agent.train = True

            if win_rate_r < 50:
                # if the previous agent beats you, train against that
                strategy_collapse(players3, agent)
                games_i += 2500

            agent.save_model("weak")

        i += 1
        games_i += 1


def strategy_collapse(players, agent):
    """
    In order to prevent strategy collapse, we ocassionally train against former version of ourself
    that beat us in evaluation
    """
    wins = 0
    for i in range(2500):
        game = Game(players)
        game.initialize()
        game.play_game()
        agent.optimize_model()

        if i % 250 == 0:
            agent.mirror_models()


def evaluate(num_games, players, idx=0, debug=False):
    """
    Evaluate the agent with the given index to count the number of wins of the particular agent
    """
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


if __name__ == "__main__":
    # train_dqn(1)
    agent = DQNAgent(False, True)
    agent.load_model("final")
    shaped_agent = DQNAgent(False, True)
    shaped_agent.load_model_from_path("../saved_models/dqn_best_recurrent_2/model_dqn_final")
    shaped_agent.eval = True
    agent.eval = True
    players = [agent, shaped_agent, agent, shaped_agent]
    players = [shaped_agent, RandomAgent(), shaped_agent, RandomAgent()]
    evaluate(1000, players, 0, False)

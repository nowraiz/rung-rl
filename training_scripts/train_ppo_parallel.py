import torch
from torch.multiprocessing import Pipe
from torch.utils.tensorboard import SummaryWriter

from rung_rl.agents.ppo.ppo_agent import PPOAgent
from rung_rl.agents.random_agent import RandomAgent
from rung_rl.game.env import RungEnv
from rung_rl.game.Game import Game

torch.multiprocessing.set_sharing_strategy('file_system')
PROCESSES = 12
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


def train_ppo(num_games, debug=False):
    agent = PPOAgent()
    agent.load_model("final")
    agent.save_model("final")
    weak_agent = PPOAgent()
    summary_writer = SummaryWriter()
    it = 0
    # dqn_agent = DQNAgent(True)
    # dqn_agent.eval = True
    win_rate_radiant = []
    games_list = []
    win_rate_dire = []
    games = []
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
        (conn1, conn2) = Pipe()
        p = RungEnv(conn2)
        processes.append(p)
        pipes.append(conn1)
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
            pipe.send("REFRESH")
            pipe.send(agent.actor.state_dict())
            pipe.send(agent.critic.state_dict())
            pipe.send("RESET")
            running[i] = True
            games += 1

        while games < CONCURRENT_GAMES:
            for i, p in enumerate(processes):
                pipe = pipes[i]
                if running[i]:
                    if (pipe.poll()):
                        # game finished
                        msg = pipe.recv()
                        assert msg == "END"
                        state_game_batch = pipe.recv()
                        action_game_batch = pipe.recv()
                        reward_game_batch = pipe.recv()
                        log_prob_game_batch = pipe.recv()
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
                        # game finished
                        msg = pipe.recv()
                        assert msg == "END"
                        state_game_batch = pipe.recv()
                        action_game_batch = pipe.recv()
                        reward_game_batch = pipe.recv()
                        log_prob_game_batch = pipe.recv()
                        state_batch += state_game_batch
                        action_batch += action_game_batch
                        reward_batch += reward_game_batch
                        log_prob_batch += log_prob_game_batch

                        running[i] = False

        total_games += games
        print(f'Total games: {total_games}')
        print(len(state_batch), len(action_batch), len(log_prob_batch), len(reward_batch))
        action_loss, value_loss, entropy = agent.optimize_model_directly(state_batch, action_batch, log_prob_batch,
                                                                         reward_batch, len(state_batch))
        it += 1
        summary_writer.add_scalar("Loss/Action", action_loss, total_games)
        summary_writer.add_scalar("Loss/Value", value_loss, total_games)
        summary_writer.add_scalar("Entropy", entropy, total_games)

        state_batch = []
        action_batch = []
        log_prob_batch = []
        reward_batch = []

        if (total_games % (CONCURRENT_GAMES * 50)) == 0:
            weak_agent.load_model("final")
            temp_players = [
                weak_agent.get_player(False),
                agent.get_player(False),
                weak_agent.get_player(False),
                agent.get_player(False)
            ]

            win_rate_self, _ = evaluate(500, temp_players, 1)

            win_rate_radiant.append(win_rate_self)
            players = [agent.get_player(False), RandomAgent(), agent.get_player(False), RandomAgent()]
            win_rate_d, _ = evaluate(500, players, 0, False)

            # games_list.append(total_games)
            # win_rate_dire.append(win_rate_d)
            # plt.plot(games_list, win_rate_radiant, win_rate_dire)
            # plt.savefig()

            summary_writer.add_scalar("WinRate/Self", win_rate_self, total_games)
            summary_writer.add_scalar("WinRate/Random", win_rate_d, total_games)
            agent.clear_experience()
            agent.save_model("final")

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
    train_ppo(10000000)
    # test()
    agent = PPOAgent(False)
    players = [agent.get_player(False), RandomAgent(), agent.get_player(False), RandomAgent()]
    evaluate(1000, players, 0, False)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()

from rung_rl.rung import Game
from rung_rl.agents import RandomAgent, HumanAgent
from rung_rl.agents import PPOAgent, save_policy, update_parameters
import numpy as np
from multiprocessing import Process

CONCURRENT_GAMES = 128

def run_game(players):
    game = Game(players)
    game.initialize()
    game.play_game()

def train(num_games, num_processes):
    model_version = 0
    print("Starting training...")
    for i in range(num_games//num_processes):
        # pool = Pool(processes=4)
        if i % (num_games//num_processes/20) == 0:
            save_policy(model_version)
            model_version += 1
        processes = []
        for j in range(num_processes):
            p = Process(target=run_game())
            p.start()
            processes.append(p)
        for p in processes:
            p.join() # wait for each game to finish
        print(i*num_processes)
    save_policy("final")
    evaluate(1000)

def train_sequential(num_games):
    agents = []
    # ai_vs_shit = [ai_vs_ai[0], RandomAgent(1), ai_vs_ai[2], RandomAgent(4)]
    model_version = 0
    for i in range(num_games//CONCURRENT_GAMES):
        agents = []
        for j in range(CONCURRENT_GAMES):
            ai_vs_shit = [PPOAgent(0), RandomAgent(1), PPOAgent(2), RandomAgent(3)]
            ai_vs_ai = [PPOAgent(0), PPOAgent(1), PPOAgent(2), PPOAgent(3)]
            if i % (num_games/20) == 0:
                save_policy(model_version)
                model_version += 1
            print(i*CONCURRENT_GAMES+j)
            if (i % 5 == 0):
                run_game(ai_vs_shit)
                agents += ai_vs_shit
            else:
                run_game(ai_vs_ai)
                agents += ai_vs_ai
            update_parameters((i*CONCURRENT_GAMES)+j,num_games)
        for agent in agents:
            agent.train()
            
    evaluate(1000)
    save_policy("final")

def evaluate(num_games, debug=False):
    print("Starting evaluation...")
    wins = 0
    r = 0
    for _ in range(num_games):
        players = [PPOAgent(0, True), RandomAgent(1), PPOAgent(2, True), RandomAgent(3)]
        game = Game(players, debug, debug)
        game.initialize()
        game.play_game()
        rewards = [players[0].rewards, players[1].rewards, players[2].rewards, players[3].rewards]
        win = int(players[0].rewards == max(rewards))
        wins += win
        r += rewards[0]
        print(rewards[0])
    print(wins, wins/num_games, r, r/num_games)

def play_game():
    players = [RandomAgent(0), HumanAgent(1), RandomAgent(2), RandomAgent(3)]
    game = Game(players, True, True)
    game.initialize()
    game.play_game()

if __name__ == "__main__":
    train_sequential(100)
from rung_rl.rung import Game
from rung_rl.agents import RandomAgent, HumanAgent
from rung_rl.agents import PPOAgent, save_policy, update_parameters
import numpy as np
from multiprocessing import Process

def run_game(rand=False):
    players = None;
    if rand:
        players = [PPOAgent(0), RandomAgent(1), PPOAgent(3), RandomAgent(4)]
    else:
        players = [PPOAgent(0), PPOAgent(1), PPOAgent(2), PPOAgent(3)]
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
    model_version = 0
    for i in range(num_games):
        print(i)
        if i % (num_games/20) == 0:
            save_policy(model_version)
            model_version += 1
        run_game(i%5 == 0)
        update_parameters(i,num_games)
    evaluate(1000)
    save_policy("final")

def evaluate(num_games, debug=False):
    print("Starting evaluation...")
    wins = 0
    for _ in range(num_games):
        players = [PPOAgent(0, True), RandomAgent(1), PPOAgent(2, True), RandomAgent(3)]
        game = Game(players, debug, debug)
        game.initialize()
        game.play_game()
        rewards = [players[0].rewards, players[1].rewards, players[2].rewards, players[3].rewards]
        win = int(players[0].rewards == max(rewards))
        wins += win
        print(rewards[0])
    print(wins, wins/num_games)

def play_game():
    players = [RandomAgent(0), HumanAgent(1), RandomAgent(2), RandomAgent(3)]
    game = Game(players, True, True)
    game.initialize()
    game.play_game()

if __name__ == "__main__":
    train_sequential(100)
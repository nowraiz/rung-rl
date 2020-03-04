from rung_rl.rung import Game
from rung_rl.agents import RandomAgent
from rung_rl.agents import PPOAgent, save_policy
import numpy as np
from multiprocessing import Process

def run_game():
    players = [PPOAgent(), PPOAgent(), PPOAgent(), PPOAgent()]
    game = Game(players)
    game.initialize()
    game.play_game()
    invalid_moves = [players[0].invalid_moves, players[1].invalid_moves, players[2].invalid_moves, players[3].invalid_moves]
    print(invalid_moves)

def train(num_games, num_processes):
    for i in range(num_games//num_processes):
        # pool = Pool(processes=4)
        if i % 4000 == 0:
            save_policy(i)
        processes = []
        for j in range(num_processes):
            p = Process(target=run_game())
            p.start()
            processes.append(p)
        for p in processes:
            p.join() # wait for each game to finish
        print(i*num_processes)
    evaluate()

def train_sequential(num_games):
    for i in range(num_games):
        if i % num_games/20 == 0:
            save_policy(i)
        run_game()
    evaluate()
    
def evaluate():
    wins = 0
    for _ in range(1000):
        players = [PPOAgent(True), RandomAgent(), PPOAgent(True), RandomAgent()]
        game = Game(players, False, False)
        game.initialize()
        game.play_game()
        rewards = [players[0].rewards, players[1].rewards, players[2].rewards, players[3].rewards]
        win = int(players[0].rewards == max(rewards))
        wins += win
        print(rewards, players[0].invalid_moves, players[2].invalid_moves)
    print(wins)

# if __name__ == "__main__":
#     train(4)
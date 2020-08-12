import random


class HumanAgent:
    def __init__(self, id):
        print("You are player", id)
        self.id = id
        self.rewards = 0

    def get_move(self, cards, hand, stack, rung, num_hand, dominating, action_mask):
        print("Your cards: ", list(zip(cards, list(range(13)))))
        move = int(input("Move: "))
        while action_mask[move] == 0:
            print("Invalid move")
            move = int(input("Move: "))
        return move

    def reward(self, val, *args):
        self.rewards += val

    def save_obs(self, *args):
        pass

    def end(self, *args):
        pass

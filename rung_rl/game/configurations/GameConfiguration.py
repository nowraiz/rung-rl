from abc import ABC, abstractmethod


class GameConfiguration(ABC):
    """
    This abstract class holds the configuration of the game such as what kind of rewards the game keeps. How many
    rounds do you need to win the game, whether ACE can win the round or not etc. which directly influence
    the rules of the game.
    """

    @abstractmethod
    def __init__(self, ace_wins, rounds_to_win, binary_rewards):
        self.ACE_WINS = ace_wins
        self.ROUNDS_TO_WIN = rounds_to_win
        self.BINARY_REWARDS = binary_rewards

from rung_rl.agents.dqn.dqn_agent import DQNAgent
from rung_rl.agents.human_agent import HumanAgent
from rung_rl.game.Game import Game

BEST_MODEL_PATH = ""


class GameSession:
    """
    This class represents a game session consisting of a complete game and the associated players. Game is progressed
    through APIs exposed by this class. Like getting move etc.
    """

    def __init__(self, players):
        self.players = players
        self.game = Game(players)

    def step(self, player, move):
        """
        Steps through the game for the given player with the given move
        """
        pass

    def get_state(self, player, oracle=False):
        """
        Returns the observation for the given player. If oracle is set, the state contains the complete
        state of the game, even the state that is invisible to the player
        """
        pass

    @staticmethod
    def create_single_player_game():
        """
        Creates an instance of the game that contains one WebAgent and three AI agents
        """
        ai_agent = DQNAgent(train=False,
                            recurrent=True)
        ai_agent.load_model_from_path(BEST_MODEL_PATH)
        ai_agent.eval = True
        return GameSession(players=[HumanAgent(0), ai_agent, ai_agent, ai_agent])

    @staticmethod
    def create_multi_player_game():
        """
        Creates an instance of the game session that contains two WebAgent and two AI agents.
        Both the WebAgents are on the same team and AI agent are on the other
        """
        ai_agent = GameSession.get_best_ai_agent()
        return GameSession(players=[HumanAgent(0), ai_agent, HumanAgent(1), ai_agent])

    @staticmethod
    def get_best_ai_agent():
        """
        Returns the best AI agent specified to be used in the production
        """
        ai_agent = DQNAgent(train=False,
                            recurrent=True)
        ai_agent.load_model_from_path(BEST_MODEL_PATH)
        ai_agent.eval = True
        return ai_agent

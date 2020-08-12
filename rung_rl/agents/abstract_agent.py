"""
This class lays out an abstract agent that describes a player in the sense of 
a card game enviornment. Any concrete implementation has to override these methods
to be able to consider an agent. 
"""


class AbstractAgent:
    """
    This function is called at the start of every time step to get the move at the
    current state. 
    """

    def get_move(self, rung_state):
        raise NotImplementedError

    """
    This function is called at the end of every time step to signal the reward
    recieved at step t
    """

    def reward(self, r, done=False):
        raise NotImplementedError

    """
    This function is called at the end of the game to signal the end of the game. 
    """

    def end(self):
        raise NotImplementedError

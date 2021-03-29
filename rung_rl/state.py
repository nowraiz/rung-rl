from rung_rl.obs import Observation

"""
Represents a single instance of state that is observable by an agent in a single timestep. This state is passed on to
the agent to get the move.

    cards: the cards in the player hand
    hand: the cards played in the current hand
    stack: the number of stacks of card beneath the current hand (signifies the number of points you get)
    rung: the rung of this instance of the game
    hand_idx: which hand is being played
    dominant: which player was dominant at the end of last hand (or the player who started this round)
    last_hand: the cards in the previous hand
    highest: which player has the highest card till now on the current hand
    last_dominant: which player was dominant at the end of second last hand
    player_id: the id of the player looking at the state
    cards_played: the cards played till this move
    action_mask: the mask of the available actions the current player can take
"""


class State:

    def __init__(self, cards, hand=None, stack=None, rung=None, hand_idx=None, dominant=None, last_hand=None, highest=None,
                 last_dominant=None, player_id=None, cards_played=None, cards_played_by=None,
                 score=None, enemy_score=None, last_turn=None, next_turn=None, has_partner_played=None, action_mask=None):
        self.cards = cards
        self.hand = hand
        self.stack = stack
        self.rung = rung
        self.hand_idx = hand_idx
        self.dominant = dominant
        self.last_hand = last_hand
        self.highest = highest
        self.last_dominant = last_dominant
        self.player_id = player_id
        self.last_turn = last_turn
        self.next_turn = next_turn
        self.score = score
        self.enemy_score = enemy_score
        self.cards_played = cards_played
        self.cards_played_by = cards_played_by
        self.has_partner_played = has_partner_played
        self.action_mask = action_mask

    """
    Returns the observation from the given state used to vectorize the state
    """

    def get_obs(self):
        return Observation(self)

    """
    Returns the mask of the actions available at the current timestep to the agent. It is used to select
    an allowed action.
    """

    def get_action_mask(self):
        return self.action_mask

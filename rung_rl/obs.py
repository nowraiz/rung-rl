from rung_rl.deck import Card,Suit
from rung_rl.utils import flatten
SUIT_SCALING = 3
FACE_SCALING = 13
STACK_SCALING = 13
HAND_SCALING = 13
RUNG_SCALING = SUIT_SCALING # rung can only be one of the four possible suits
ID_SCALING = 3 # the players (4 actually)
class Observation():
    """
    Observation represents an observation of the environment at a single timestep
    """
    def __init__(self,
                cards : [Card],
                hand : [Card], 
                stack : int, 
                rung : Suit,
                num_hand : int,
                dominating : int,
                last_hand : [Card],
                id : int,
                cards_seen  = None):
        self.cards : [Card] = cards # cards in agents hand
        self.hand : [Card] = hand # the current hand of the game
        self.stack : int = stack # the stack of hands beneath the current hand
        self.rung : Suit = rung # rung for this game
        self.num_hand : int = num_hand # number of the current hand
        self.dominating : int = dominating # the player id of the person dominating
        self.id : int = id # my player id
        self.last_hand = last_hand
        self.cards_seen : [int] = cards_seen
        self.obs_vector = None
        self.build_vector()
    
    def get(self):
        """
        Returns the observation in a vector
        """
        return self.obs_vector

    def build_vector(self):
        """ 
        Builds the observation vector from the data points given
        doing rescaling as needed
        """
        self.obs_vector = \
            flatten([self.embed_card(card) for card in self.cards]) + \
            flatten([self.embed_card(card) for card in self.hand]) + \
            flatten([self.embed_card(card) for card in self.last_hand]) + \
            self.embed_suit(self.rung) + \
            self.embed_player(self.id) + \
            self.embed_player(self.dominating) + \
            [self.stack, self.num_hand]


    def scale_card(self,card : Card):
        """
        returns the scaled representation of the card's face
        and suit
        """
        return ((card.face.value - 1), card.suit.value)

    def embed_card(self, card: Card):
        x = [0 for _ in range(17)]
        if card is not None:
            x[card.face.value - 2] = 1
            x[card.suit.value + 13 - 1] = 1
        return x
    def embed_suit(self, suit):
        x = [0 for _ in range(4)]
        x[suit.value-1] = 1
        return x
    def embed_player(self, player):
        x = [0 for _ in range(4)]
        if player is not None:
            x[player] = 1
        return x


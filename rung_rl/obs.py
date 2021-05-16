from rung_rl.deck import Card, Suit
from rung_rl.utils import flatten

SUIT_SCALING = 3
FACE_SCALING = 13
STACK_SCALING = 13
HAND_SCALING = 13
RUNG_SCALING = SUIT_SCALING  # rung can only be one of the four possible suits
ID_SCALING = 3  # the players (4 actually)


class Observation:
    """
    Observation represents an observation of the environment at a single timestep
    """

    def __init__(self,
                 state):
        self.cards: [Card] = state.cards  # cards in every agent hand
        self.id: int = state.player_id  # my player id
        self.hand: [Card] = state.hand  # the current hand of the game
        self.hand_played_by = state.hand_played_by # the players of have played the cards in the hand
        self.stack: int = state.stack  # the stack of hands beneath the current hand
        self.rung: Suit = state.rung  # rung for this game
        self.num_hand: int = state.hand_idx  # number of the current hand
        self.hand_idx: int = state.hand_place # the index of the player in the current hand
        self.dominating: int = state.dominant  # the player id of the person dominating
        self.last_hand = state.last_hand
        self.cards_seen: [int] = state.cards_played
        self.cards_played_by = state.cards_played_by
        self.highest = state.highest  # the player who has the highest card right now on the table
        self.highest_card =state.highest_card # the highest card in the current hand
        self.higher_cards = state.higher_cards # which of the players card are potentially higher if played
        self.winning_round = state.winning_round # if the current highest played can win round if not crossed
        self.last_dominant = state.last_dominant  # the player who was dominant in last to last round
        self.partner = (self.id + 2) % 4 if self.id else None
        self.next_turn = -1 if state.next_turn == None else state.next_turn
        self.last_turn = -1 if state.last_turn == None else state.last_turn
        self.score = state.score
        self.enemy_score = state.enemy_score
        self.has_partner_played = state.has_partner_played
        self.action_mask = state.action_mask
        self.obs_vector = None
        self.rung_vector = None
        # self.build_vector()

    def get(self):
        """
        Builds and returns the observation in a vector
        """
        assert self.hand != None # just to make sure we are not using state that is meant for building run state
        self.build_vector()
        return self.obs_vector

    def get_rung(self):
        """
        Build and returns the initial observation for the selection of rung in a vector
        """
        self.build_rung_vector()
        return self.rung_vector

    def build_rung_vector(self):
        """
        Builds the rung observation vector from the initial data points
        """
        self.rung_vector = \
            flatten([self.embed_card(card) for card in self.cards[self.id] if card])
    
    def build_vector(self):
        """ 
        Builds the observation vector from the data points given
        doing rescaling as needed
        """
        self.obs_vector = \
            flatten([self.embed_card(card) for card in self.cards[self.id]]) + \
            self.embed_hand_cards() + \
            self.embed_cards_seen() + \
            self.embed_card(self.highest_card) + self.embed_player(self.highest) + \
            self.embed_suit(self.rung) + \
            self.embed_player(self.id) + \
            self.embed_player(self.dominating) + \
            self.embed_player(self.last_dominant) + \
            self.embed_player(self.highest) + \
            self.higher_cards + \
            self.action_mask + \
            [1 if self.winning_round else 0] + \
            self.embed_player(self.partner) + \
            self.embed_player(self.last_turn) + \
            self.embed_player(self.next_turn) + \
            [self.score, self.enemy_score, self.has_partner_played, self.stack, self.num_hand]

        # increase length by 16 for hand
        # increase by 17 + 4 for the highest card
        # increase by 13 for higher cards
        # increase by 1 for winning_round
        # increase by 13 for action_mask : NOT REQUIRED

    def get_oracle_vector(self):
        """
        Returns the observation of the complete game (perfect information)
        that should only be used by the oracle
        """
        self.oracle_vector = \
            flatten([self.embed_card(card) for card in self.cards[0]]) + \
            flatten([self.embed_card(card) for card in self.cards[1]]) + \
            flatten([self.embed_card(card) for card in self.cards[2]]) + \
            flatten([self.embed_card(card) for card in self.cards[3]]) + \
            self.embed_hand_cards() + \
            self.embed_card(self.highest_card) + self.embed_player(self.highest) + \
            self.embed_suit(self.rung) + \
            self.higher_cards + \
            [1 if self.winning_round else 0] + \
            self.embed_player(self.last_turn) + \
            self.embed_player(self.next_turn) + \
            self.embed_player(self.id) + \
            self.embed_player(self.dominating) + \
            self.embed_player(self.last_dominant) + \
            self.embed_player(self.partner) + \
            [self.score, self.enemy_score, self.has_partner_played, self.stack, self.num_hand, self.hand_idx]
        
        return self.oracle_vector
    
    def get_oracle_rung_vector(self):
        """
        Returns the starting cards for all the players for the oracle
        to select the optimal rung (hopefully)
        """
        self.oracle_rung_vector = \
            flatten([self.embed_card(card) for card in self.cards[0] if card]) + \
            flatten([self.embed_card(card) for card in self.cards[1] if card]) + \
            flatten([self.embed_card(card) for card in self.cards[2] if card]) + \
            flatten([self.embed_card(card) for card in self.cards[3] if card]) + \
            self.embed_player(self.id)
        
        return self.oracle_rung_vector

    def scale_card(self, card: Card):
        """
        returns the scaled representation of the card's face
        and suit
        """
        return (card.face.value - 1), card.suit.value

    def embed_hand_cards(self):
        embedding = []
        for i, card in enumerate(self.hand):
            embedding += self.embed_card(card) + self.embed_player(self.hand_played_by[i])
        return embedding
        
    def embed_cards_seen(self):
        embedding = []
        assert len(self.cards_played_by) == len(self.cards_seen)
        for i in range(len(self.cards_seen)):
            card = self.embed_card(self.cards_seen[i])
            card = card + self.embed_player(self.cards_played_by[i])
            embedding = embedding + card
        return embedding
    def embed_card(self, card: Card):
        x = [0 for _ in range(17)]
        if card is not None:
            x[card.face.value - 2] = 1
            x[card.suit.value + 13 - 1] = 1
        return x

    def embed_player_card(self, card: Card, player):
        return self.embed_card(card) + self.embed_player(player)

    def embed_suit(self, suit):
        x = [0 for _ in range(4)]
        if suit is not None:
            x[suit.value - 1] = 1
        return x

    def embed_player(self, player):
        x = [0 for _ in range(4)]
        if player is not None:
            x[player] = 1
        return x

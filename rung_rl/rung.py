from rung_rl.deck import Deck
import random
NUM_PLAYERS = 4 # the number of players is fixed for RUNG i.e. 4
NUM_TEAMS = 2 # the number of teams is fixed for RUNG i.e. 2
REWARD_SCALE = 26
class Game():
    """
    A round represents a round of the whole game. The rules are enforced at all times (it can not be in 
    an invalid state)
    """

    def __init__(self, players, render=False, debug=False):
        self.hand_idx = 0 # the index of the next hand 
        self.hands = 0 # the number of hands played
        self.stack = 0 # the number of hands stacked right now
        self.render = render # whether to render the intermediate stages
        self.debug = debug # whether to print intermediate debug information about the game
        self.deck = Deck() # start with a standard deck of standard playing cards
        self.hand = [None, None, None, None] # a hand represents the playing hand
        self.hand_player = [None, None, None, None] # the index of the player who played this card on the hand
        self.current_player = random.randint(0,3) # the index of the player whose turn it is
        self.scores = [0 for _ in range(4)] # the scores for the teams
        self.player_cards = [[None for _ in range(13)] for _ in range(4)]
        self.players = players
        self.rung = None # the rung for this round
        self.dominant_player = None
        self.done = False
    def to_string(self, cards):
        return list(map(str, cards));
    def draw_card_from_deck(self):
        """
        Draw the top card of the deck
        """
        return self.deck.pop()

    def populate_initial(self):
        """
        Populate 5 cards to each player in the round robin fashion from the deck
        """
        for i in range(0, 20,5):
            k = i // 5
            for j in range(0, 5):
                self.player_cards[k][j] = self.draw_card_from_deck()

    def populate_remaining_cards(self):
        """
        Populate the remaining cards on the deck to the players in a round-robin fashion of 4 each
        """
        for it in range(2):
            for i in range(0, 16,4):
                k = i // 4
                for j in range(0,4):
                    self.player_cards[k][j+5+it*4] = self.draw_card_from_deck()
        

    def initialize(self):
        """
        Initialize the players card and sets the rung as directed by the player and populates the
        remaining cards
        """
        assert(self.rung == None)
        # initialize the game and set the rung
        self.populate_initial()
        player = self.players[self.current_player]
        move = player.get_move(self.player_cards[self.current_player], self.hand)
        while not self.valid_move(move, self.current_player):
            player.reward(-0.001, True)
            move = player.get_move(self.player_cards[self.current_player], self.hand)
        card = self.peek_card(move, self.current_player)
        self.rung = card.suit
        self.DEBUG("Setting rung equal to", self.rung)
        self.populate_remaining_cards()
        assert(self.deck.length() == 0)

        # debug
        for i in range(4):
            self.DEBUG(self.to_string(self.player_cards[i]))

    def broadcast(self):
        """
        Broadcast the hand information to the agents who do not have
        their turn right now
        """
        for i in range(4):
            if i != self.current_player:
                self.players[i].save_obs(self.hand)

    def play_hand(self):
        """
        Play a hand of rung. Goes around the players asking them for the moves at each turn
        """
        assert(not self.done)
        self.DEBUG("Starting hand", self.hands + 1)
        player_idx = [0,0,0,0]
        for i in range(4):
            player = self.players[self.current_player]
            move = player.get_move(self.player_cards[self.current_player], self.hand)
            while not self.valid_move(move, self.current_player):
                player.reward(-0.001, True)
                move = player.get_move(self.player_cards[self.current_player], self.hand)
            self.hand[self.hand_idx] = self.draw_card(move, self.current_player)
            player_idx[self.hand_idx] = self.current_player
            self.hand_idx += 1
            self.broadcast() # broadcast the hand information to every other player
            self.current_player = (self.current_player + 1) % 4
            self.DEBUG(self.to_string(self.hand))

        self.stack += 1 # increase the stack count
        self.hands += 1
        idx = self.highest_card_index() # get the highest card
        dominant = player_idx[idx] # get the player who played the highest card
        dominant_player = self.players[dominant]
        if self.hands == 13 or (dominant == self.dominant_player and self.hands > 2):
            winner1 = dominant
            winner2 = (dominant + 2) % 4
            reward = self.stack / REWARD_SCALE
            self.scores[winner1] += reward
            self.scores[winner2] += reward
            if max(self.scores) > 6:
                # game is done
                self.done = 1
                reward += 13 / REWARD_SCALE
            for i, player in enumerate(self.players):
                if i == winner1 or i == winner2:
                    player.reward(reward)
                else:
                    player.reward(-reward)
            self.stack = 0
            if self.hands == 13:
                self.done = 1
        else:
            for player in self.players:
                player.reward(0)
            self.dominant_player = dominant
        # clear the hand and set the new next player
        self.hand = [None for _ in range(4)]
        self.hand_idx = 0
        self.current_player = self.dominant_player
        self.DEBUG("Ending hand", self.hands)

    def play_game(self):
        for i in range(13):
            if self.game_over():
                self.end_game()
                return
            self.play_hand()
        self.end_game()
    def end_game(self):
        """
        Signal all players that it is an end of the game
        """
        assert(self.done)
        for player in self.players:
            player.end()
        self.DEBUG("Game ended")

    def draw_card(self, move, player):
        """
        Draws the ith card from the player cards and returns it
        """
        # gets the ith card from the player and returns it
        card = self.player_cards[player][move]
        self.player_cards[player][move] = None
        return card
    def peek_card(self, move, player):
        """
        Peeks the ith card of the player cards
        """
        return self.player_cards[player][move]
    def empty_hand(self):
        """
        Checks if it is a start of the hand (empty hand)
        """
        return self.hand_idx == 0
    def valid_move(self, move, player):
        """
        Check if it is a valid move for the given player
        """
        card = self.peek_card(move, player)
        return card is not None and \
                ( self.empty_hand() or \
                self.hand[0].suit == card.suit or \
                not self.has_suit(player) )
    
    def has_suit(self, player):
        """
        Check if the given player has the current suit on the hand
        """
        suit = self.hand[0].suit
        for card in self.player_cards[player]:
            if card and card.suit == suit:
                return True
        return False

    def game_over(self):
        """
        Checks if the game has ended
        """
        return self.done
    def highest_card_index(self):
        """
        Checks which of the card on the current hand is the highest based on rules of rung
        """
        top = self.hand[0]
        index = 0
        for i, card in enumerate(self.hand[1:]):
            if (top.suit == card.suit and card.face > top.face) or card.suit == self.rung:
                top = card
                index = i
        # print(str(top))
        return index
    def DEBUG(self, *args):
        if self.debug:
            print(*args)
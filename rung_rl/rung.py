from rung_rl.agents.dqn.dqn_agent import DQNAgent
from rung_rl.deck import Deck, Card
from rung_rl.deck import Suit
import random

NUM_PLAYERS = 4  # the number of players is fixed for RUNG i.e. 4
NUM_TEAMS = 2  # the number of teams is fixed for RUNG i.e. 2
REWARD_SCALE = 65



class Game():
    """
    A round represents a round of the whole game. The rules are enforced at all times (it can not be in 
    an invalid state)
    """

    def __init__(self, players, render=False, debug=False):
        self.hand_idx = 0  # the index of the next hand
        self.hands = 0  # the number of hands played
        self.stack = 0  # the number of hands stacked right now
        self.render = render  # whether to render the intermediate stages
        self.debug = debug  # whether to print intermediate debug information about the game
        self.deck = Deck()  # start with a standard deck of standard playing cards
        self.hand = [None, None, None, None]  # a hand represents the playing hand
        self.hand_player = [None, None, None, None]  # the index of the player who played this card on the hand
        self.current_player = random.randint(0, 3)  # the index of the player whose turn it is
        self.scores = [0 for _ in range(4)]  # the scores for the teams
        self.player_cards = [[None for _ in range(13)] for _ in range(4)]
        self.players = players
        self.rung = None  # the rung for this round
        self.dominant = None
        self.last_dominant = None
        self.done = False
        self.last_hand = [None, None, None, None]

    def sort_cards(self, cards):
        return sorted(cards, key=lambda card: (card.suit.value - 1) * 13 + card.face.value - 2 if card else 0)

    def to_string(self, cards: [Card]):
        return list(map(str, cards))

    def draw_card_from_deck(self):
        """
        Draw the top card of the deck
        """
        return self.deck.pop()

    def populate_initial(self):
        """
        Populate 5 cards to each player in the round robin fashion from the deck
        """
        for i in range(0, 20, 5):
            k = i // 5
            for j in range(0, 5):
                self.player_cards[k][j] = self.draw_card_from_deck()

    def populate_remaining_cards(self):
        """
        Populate the remaining cards on the deck to the players in a round-robin fashion of 4 each
        """
        for it in range(2):
            for i in range(0, 16, 4):
                k = i // 4
                for j in range(0, 4):
                    self.player_cards[k][j + 5 + it * 4] = self.draw_card_from_deck()

    def initialize(self):
        """
        Initialize the players card and sets the rung as directed by the player and populates the
        remaining cards
        """
        assert (self.rung == None)
        # initialize the game and set the rung
        self.populate_initial()
        # player = self.players[self.current_player]
        # move = player.get_move(self.player_cards[self.current_player], self.hand, self.stack, self.rung.value if self.rung else 0, self.hands, self.action_mask(self.current_player))
        # while not self.valid_move(move, self.current_player):
        # player.reward(0)
        # move = player.get_move(self.player_cards[self.current_player], self.hand)
        # assert(self.valid_move(move, self.current_player))
        # card = self.peek_card(move, self.current_player)
        self.rung = random.choice([suit for suit in Suit])
        self.DEBUG("Rung : ", self.rung)
        self.populate_remaining_cards()
        assert (self.deck.length() == 0)

        self.dump_all_cards()

    def broadcast(self):
        """
        Broadcast the hand information to the agents who do not have
        their turn right now
        """
        for i in range(4):
            self.players[i].save_obs(self.hand)

    def play_hand(self):
        """
        Play a hand of rung. Goes around the players asking them for the moves at each turn
        """
        assert (not self.done)
        self.DEBUG("Starting hand", self.hands + 1)
        player_idx = [0, 0, 0, 0]
        for i in range(4):
            if self.hand_idx == 0:
                highest = None
            else:
                highest_idx = self.highest_card_index()
                highest = player_idx[highest_idx]

            self.DEBUG("Player", self.current_player, " turn")
            player = self.players[self.current_player]
            move = player.get_move(self.player_cards[self.current_player], self.hand, self.stack,
                                   self.rung if self.rung else 0, self.hands + 1, self.dominant, self.last_hand,
                                   highest, self.action_mask(self.current_player), self.last_dominant)

            move = move.item()
            # player.reward(-0.001, True)
            # move = player.get_move(self.player_cards[self.current_player], self.hand)
            self.DEBUG(self.dump_cards(self.player_cards[self.current_player]), str(self.player_cards[self.current_player][move]))
            assert (self.valid_move(move, self.current_player))
            self.hand[self.hand_idx] = self.draw_card(move, self.current_player)
            player_idx[self.hand_idx] = self.current_player
            self.hand_idx += 1
            self.current_player = (self.current_player + 1) % 4
            self.DEBUG(self.to_string(self.hand))

        self.broadcast()  # broadcast the hand information to every other player
        self.stack += 1  # increase the stack count
        self.hands += 1
        idx = self.highest_card_index()  # get the highest card
        dominant = player_idx[idx]  # get the player who played the highest card
        if self.hands == 13 or (dominant == self.dominant and self.hands > 2):
            winner1 = dominant
            winner2 = (dominant + 2) % 4
            reward = self.stack / REWARD_SCALE
            # reward = 0
            self.scores[winner1] += self.stack
            self.scores[winner2] += self.stack
            if max(self.scores) > 6:
                # game is done
                self.done = True
                reward += 52 / REWARD_SCALE
                # reward = 1
                self.DEBUG("WINNER: ", winner1, winner2)
            for i, player in enumerate(self.players):
                if i == winner1 or i == winner2:
                    player.reward(reward, self.done)
                else:
                    player.reward(-reward, self.done)
            self.stack = 0

        else:
            for player in self.players:
                player.reward(0)
        self.last_dominant = self.dominant
        self.dominant = dominant
        # clear the hand and set the new next player
        self.last_hand = self.hand
        self.hand = [None for _ in range(4)]
        self.hand_idx = 0
        self.current_player = self.dominant
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
        assert (self.done)
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
               (self.empty_hand() or \
                self.hand[0].suit == card.suit or \
                not self.has_suit(player))

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
        for i, card in enumerate(self.hand):
            if self.higher_card(top, card):
                top = card
                index = i
        # print(str(top))
        return index

    def DEBUG(self, *args):
        if self.debug:
            print(*args)

    def higher_card(self, card1: Card, card2: Card) -> bool:
        """
        Returns if card2 is higher than card1 according to rung rules
        """
        if card2 is None:
            return False
        return (card1.suit == card2.suit and card2.face.value > card1.face.value) or \
               (card1.suit != self.rung and card2.suit == self.rung)

    def action_mask(self, player):
        """
        Returns the mask for the available actions for a player
        """
        if self.hand_idx == 0 or not self.has_suit(player):
            return [1 if card else 0 for card in self.player_cards[player]]
        else:
            return [1 if card and card.suit == self.hand[0].suit else 0 for card in self.player_cards[player]]

    def dump_all_cards(self):
        for player in range(len(self.players)):
            self.DEBUG(self.to_string(self.sort_cards(self.player_cards[player])))

    def dump_cards(self, cards):
        self.DEBUG(self.to_string(self.sort_cards(cards)))

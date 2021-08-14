import random

from rung_rl.deck import Deck, Card
from rung_rl.deck import Suit
from rung_rl.state import State

NUM_PLAYERS = 4  # the number of players is fixed for RUNG i.e. 4
NUM_TEAMS = 2  # the number of teams is fixed for RUNG i.e. 2
REWARD_SCALE = 56


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
        self.cost = [0, 0, 0, 0]  # the cost of the card played in the hand by each player
        self.hand_player = [None, None, None, None]  # the index of the player who played this card on the hand
        self.current_player = None  # the index of the player whose turn it is
        self.hand_started_by = None  # the person who started the hand
        self.scores = [0 for _ in range(4)]  # the scores for the teams
        self.player_cards = [[None for _ in range(13)] for _ in range(4)]
        self.players = players
        self.rung = None  # the rung for this round
        self.dominant = None
        self.last_dominant = None
        self.done = False
        self.previous_winner = None
        self.last_hand = [None, None, None, None]
        self.last_hand_played_by = [None, None, None, None]
        self.cards_played = [None for _ in range(52)]  # the cards played till now
        self.cards_played_by = [None for _ in range(52)]  # which player played the card
        self.cards_played_idx = 0

    def sort_all_cards(self):
        for i in range(4):
            self.player_cards[i] = self.sort_cards(self.player_cards[i])

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

    def initialize(self, player=None):
        """
        Initialize the players card and sets the rung as directed by the player and populates the
        remaining cards
        """
        assert (self.rung == None)
        # initialize the game and set the rung
        self.populate_initial()
        # toss for the player that selects rung
        # if we have a winner already declared. Make that player choose instead
        toss = player if player is not None else random.randint(0, 3)
        self.previous_winner = toss
        suits = [suit for suit in Suit]
        winner = self.players[toss]
        self.DEBUG(str(self.player_cards[toss]))
        move = winner.get_rung(State(self.player_cards, toss, action_mask=self.rung_action_mask()), toss)
        move = move.item()
        assert move >= 0 and move <= 3
        self.rung = suits[move]
        self.current_player = toss  # the player who selects rung starts the game
        self.DEBUG("Rung: {}, Selected By: {}".format(str(self.rung), toss))
        # player = self.players[self.current_player]
        # move = player.get_move(self.player_cards[self.current_player], self.hand, self.stack, self.rung.value if self.rung else 0, self.hands, self.action_mask(self.current_player))
        # while not self.valid_move(move, self.current_player):
        # player.reward(0)
        # move = player.get_move(self.player_cards[self.current_player], self.hand)
        # assert(self.valid_move(move, self.current_player))
        # card = self.peek_card(move, self.current_player)
        # self.rung = random.choice([suit for suit in Suit])
        # self.DEBUG("Rung : ", self.rung)
        self.populate_remaining_cards()
        self.sort_all_cards()
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
        player_idx = [None, None, None, None]
        self.hand_started_by = self.current_player
        for i in range(4):
            if self.hand_idx == 0:
                highest = None
                highest_card = None
            else:
                highest_idx = self.highest_card_index()
                highest = player_idx[highest_idx]
                highest_card = self.hand[highest_idx]

            self.DEBUG("Player", self.current_player, " turn")
            player = self.players[self.current_player]

            state = State(self.player_cards, self.current_player, self.hand, player_idx, self.stack,
                          self.rung if self.rung else None, self.hands + 1, i,
                          self.dominant, self.last_hand, self.last_hand_played_by,
                          highest, self.last_dominant, self.cards_played,
                          self.cards_played_by, self.scores[self.current_player],
                          self.scores[self.next_player()],
                          highest_card, self.higher_cards(highest_card),
                          self.last_dominant == highest,  # if the current top card is winning round
                          None if self.hand_idx == 0 else self.prev_player(),
                          None if self.hand_idx == 3 else self.next_player(),
                          int(self.partner_player() in player_idx),
                          self.action_mask(self.current_player))

            move = player.get_move(state)

            move = move.item()
            # player.reward(-0.001, True)
            # move = player.get_move(self.player_cards[self.current_player], self.hand)
            self.dump_cards(self.player_cards[self.current_player])
            self.DEBUG(str(self.player_cards[self.current_player][move]))

            valid = self.valid_move(move, self.current_player)
            if not valid:
                print(move, self.current_player, self.player_cards[self.current_player],
                      self.action_mask(self.current_player))
            assert valid
            card = self.draw_card(move, self.current_player)
            cost = self.card_cost(card)
            self.cost[self.current_player] = cost  # the cost of the card for the player
            self.hand[self.hand_idx] = card  # the hand index
            player_idx[self.hand_idx] = self.current_player  # the card played index
            # add the card to the cards seen
            self.cards_played[self.cards_played_idx] = self.hand[self.hand_idx]
            self.cards_played_by[self.cards_played_idx] = self.current_player
            self.cards_played_idx += 1

            # advance the hand and the player
            self.hand_idx += 1
            self.current_player = self.next_player()
            self.DEBUG(self.to_string(self.hand))
            # self.DEBUG("\n")

        # self.broadcast()  # broadcast the hand information to every other player
        self.stack += 1  # increase the stack count
        self.hands += 1
        idx = self.highest_card_index()  # get the highest card index
        highest_card = self.hand[idx]
        dominant = player_idx[idx]  # get the player who played the highest card
        # ADDED ACE RULE TO SEE WHAT CHANGES
        if self.hands == 13 or ((dominant == self.dominant and self.hands > 2)):
            winner1 = dominant
            winner2 = (dominant + 2) % 4
            reward = self.stack / REWARD_SCALE
            # reward = 0
            self.scores[winner1] += self.stack
            self.scores[winner2] += self.stack
            self.DEBUG("Total Rounds for {}, {}: {}".format(winner1, winner2, self.scores[winner1]))
            if max(self.scores) > 6:
                # game is done
                self.done = True
                reward += 43 / REWARD_SCALE
                # reward = 1 # binary reward
                # reward = 1
                self.DEBUG("WINNER: ", winner1, winner2)
            # IMPORTANT: Rewards players in the order that they played
            i = self.hand_started_by
            c = 0
            while c < 4:
                player = self.players[i]
                if i == winner1 or i == winner2:
                    player.reward(reward - self.cost[i], i, self.done)
                else:
                    player.reward(-reward - self.cost[i], i, self.done)
                c += 1
                i = self.next_player(i)
            self.stack = 0

        else:
            # important: reward players in the order that they played because of
            # synchronization in multiprocessing
            i = self.hand_started_by
            c = 0
            while c < 4:
                player = self.players[i]
                # the reward is just the cost of the cards played
                # by each player
                player.reward(-self.cost[i], i)
                c += 1
                i = self.next_player(i)
        self.last_dominant = self.dominant
        self.dominant = dominant
        # clear the hand and set the new next player
        self.last_hand = self.hand
        self.last_hand_played_by = player_idx
        self.cost = [0, 0, 0, 0]
        self.hand = [None for _ in range(4)]
        self.hand_idx = 0
        self.current_player = self.dominant
        if self.stack == 0:
            self.dominant = None
        self.DEBUG("Ending hand", self.hands, "\n")

    def play_game(self):
        for i in range(13):
            if self.game_over():
                w = self.end_game()
                return w
            self.play_hand()
        return self.end_game()

    def end_game(self):
        """
        Signal all players that it is an end of the game
        and return the winners of the game
        """
        assert (self.done)
        for i, player in enumerate(self.players):
            player.end(self.scores[i] > 6, i)
        self.DEBUG("Game ended")
        if self.previous_winner is not None and self.scores[self.previous_winner] > 6:
            return self.previous_winner, self.partner_player(self.previous_winner)
        winner = (self.previous_winner + 1) % NUM_PLAYERS
        return winner, self.partner_player(winner)

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

    def higher_cards(self, highest_card):
        """
        Returns which of the current player card can be higher than the
        highest card *if* played
        """
        action_mask = self.action_mask(self.current_player)  # allowed moves
        if highest_card is None:
            # every playable card card you play will be a higher card
            return action_mask[:-4]

        higher_cards = []
        for i, card in enumerate(self.player_cards[self.current_player]):
            # check which of the cards are higher than the current highest card
            higher_cards.append(1 if self.higher_card(highest_card, card) and action_mask[i] else 0)
        return higher_cards

    def action_mask(self, player):
        """
        Returns the mask for the available actions for a player
        """
        if self.hand_idx == 0 or not self.has_suit(player):
            return [1 if card else 0 for card in self.player_cards[player]] + [0 for _ in range(4)]
        else:
            return [1 if card and card.suit == self.hand[0].suit else 0
                    for card in self.player_cards[player]] + [0 for _ in range(4)]

    def rung_action_mask(self):
        """
        Returns the mask for the available moves for a player
        during the selection of rung. 
        """
        return [0 for _ in range(13)] + [1 for _ in range(4)]

    def next_player(self, player=None):
        """
        Returns the idx of the next player in the circle of players.
        If no player is given, we assume its the current player
        """
        if player is None:
            player = self.current_player
        return (player + 1) % 4

    def prev_player(self, player=None):
        """
        Returns the idx of the previous player int he circle of players.
        If no player is given, we assume its the current player
        """
        if player is None:
            player = self.current_player
        return (player - 1) % 4

    def partner_player(self, player=None):
        """
        Returns the idx of the partner in the circle of players,
        If no player is given,it assumes we are talking about the 
        current player
        """
        if player is None:
            player = self.current_player
        return (player + 2) % 4

    def dump_all_cards(self):
        for player in range(len(self.players)):
            self.DEBUG(self.to_string(self.player_cards[player]))

    def dump_cards(self, cards):
        self.DEBUG(self.to_string(cards))

    def get_state(self):
        return

    def card_cost(self, card: Card):
        """
        Returns the cost of the card which is used to calculate
        the negative reward propotional to the cost
        TODO: Scale the cost correctly to shape the reward
        """
        cost = (card.face.value - 1) / 13
        if card.suit == self.rung:
            cost = cost * 2
        return cost / 100

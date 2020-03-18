from enum import IntEnum
from numpy.random import permutation

class Suit(IntEnum):
    """
    Suit represents one of the possible suit of a playing card.
    """
    SPADES = 1
    DIAMONDS = 2
    CLUBS = 3
    HEARTS = 4

class Face(IntEnum):
    """
    Face of card represents the value of the playing card.
    """
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

class Card():
    """
    A card represents a single card from a standard deck of card.
    It has simply a face and suit.
    """
    def __init__(self, face, suit):
        self.face : Face = face
        self.suit : Suit = suit

    def __str__(self):
        face = None
        if self.face < 11:
            face = str(self.face.value)
        else:
            face = str(self.face.name[0])
        return face + str(self.suit.name[0])

    def __repr__(self):
        return str(self)
    def to_int(self):
        return (self.face.value, self.suit.value)

class Deck():
    """
    A deck represents a standard deck of 52 cards
    """
    def __init__(self):
        self.cards = []
        self.populate_cards()
        self.shuffle_cards()
        self.shuffle_cards()
        self.shuffle_cards()
    def populate_cards(self):
        for i in Suit:
            for j in Face:
                self.cards.append(Card(j, i))
        assert(self.length() == 52)
            
    def shuffle_cards(self):
        self.cards = list(permutation(self.cards))
        pass # TODO:
    
    def pop(self):
        return self.cards.pop()
            
    def length(self):
        return len(self.cards)
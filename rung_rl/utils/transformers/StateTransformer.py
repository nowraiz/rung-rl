from rung_rl.deck import Card
from rung_rl.state import State


class StateTransformer:
    """
    Serializes the state of the given rung game to a dictionary holding JSON-like contents of the state
    as specified by the JSON schema
    """

    @staticmethod
    def create_state_dict(state: State):
        """
        Creates the state dict based on the schema
        """
        state_dict = {
            "stack": state.stack,
            "score_cyborg": state.score if state.player_id % 2 == 0 else state.enemy_score,
            "score_ai": state.score if state.player_id % 2 == 1 else state.enemy_score,
            "round": state.hand + 1,
            "rung": str(state.rung),
            "hand": []
        }
        # state_dict["player_id"] = player_id # done by the server
        for idx in range(4):
            card_dict = {}
            card = state.hand[idx]
            if card is not None:
                card_dict["player"] = state.hand_played_by[idx]
                card_dict["face"] = str(card.face)
                card_dict["suit"] = str(card.suit)
            state_dict["hand"].append(card_dict)

        state_dict["hand_idx"] = state.hand_idx
        state_dict["hand_started_by"] = None if state.hand_idx == 0 else state.hand_played_by[0]
        state_dict["done"] = state.done
        state_dict["winner"] = state.winner
        state_dict["current_player"] = state.player_id
        state_dict["player_cards"] = []

        for player_idx in range(4):
            player_dict = {"cards": []}
            for card_idx in range(13):
                card = state.cards[player_idx][card_idx]
                player_dict["cards"].append(card)
            state_dict["player_cards"].append(player_dict)

    @staticmethod
    def create_card_dict(card: Card, player_idx, card_idx, state):
        card_dict = {
            "idx": card_idx,
            "played": True,
            "playable": state.player_id == player_idx and state.action_mask[card_idx]
        }

        if card:
            card_dict["face"] = str(card.face)
            card_dict["suit"] = str(card.suit)
            card_dict["played"] = False

    @staticmethod
    def specialize_state_dict(state_dict, player_id):
        """
        Specializes the general state dictionary to the specific player id given by
        hiding the information that the player does not have information about
        """
        state_dict["player_id"] = player_id
        for player_idx in range(4):
            state_dict["player_cards"][player_idx]["visible"] = player_idx == player_id
        return state_dict

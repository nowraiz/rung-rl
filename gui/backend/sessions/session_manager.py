from .session import GameSession


class SessionManager:
    """
    This class handled the game sessions that are created for the frontend. This stores the game objects and the
    players. And is responsible for bookkeeping of the current sessions.
    """

    def __init__(self):
        self.single_player_game_sessions = {}
        self.multi_player_game_sessions = {}
        self.waiting_player = None

    def create_single_player_game(self, session_id):
        if session_id in self.single_player_game_sessions:
            # player already in an active game.
            raise AssertionError("Game for this session already exists")
        game_session = GameSession.create_single_player_game()
        self.single_player_game_sessions[session_id] = game_session
        return game_session

    def retrieve_single_player_session(self, session_id):
        if session_id not in self.single_player_game_sessions:
            # Game does not exist
            raise AssertionError("Game does not exist")
        return self.single_player_game_sessions[session_id]

    def create_multiplayer_game(self, session_id_1, session_id_2):
        if {session_id_1, session_id_2} in self.multi_player_game_sessions \
                or session_id_1 in self.single_player_game_sessions \
                or session_id_2 in self.single_player_game_sessions:
            raise AssertionError("One or more of the players is already in a game")
        game_session = GameSession.create_multi_player_game()
        self.multi_player_game_sessions[{session_id_1, session_id_2}] = game_session
        return game_session

    def retrieve_multi_player_session(self, session_id_1, session_id_2):
        if not {session_id_1, session_id_2} in self.multi_player_game_sessions:
            raise AssertionError("Game for the given players not found")
        return self.multi_player_game_sessions[{session_id_1, session_id_2}]

    def add_to_queue(self, session_id):
        """
        Adds the player with the session id to the queue for multi player queue. If the player is
        matched instantly, returns true
        """
        if self.waiting_player:
            return
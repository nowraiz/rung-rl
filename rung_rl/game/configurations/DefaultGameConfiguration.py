from rung_rl.game.configurations.GameConfiguration import GameConfiguration


class DefaultGameConfiguration(GameConfiguration):

    def __init__(self):
        super().__init__(ace_wins=False,
                         rounds_to_win=7,
                         binary_rewards=False)

from torch.multiprocessing import Pipe, Process, Queue
from rung_rl.agents.ppo.ppo_agent import PPOAgent
from rung_rl.rung import Game

from rung_rl.state import State

NUM_PLAYERS = 4  # the number of players is fixed for RUNG i.e. 4
NUM_TEAMS = 2  # the number of teams is fixed for RUNG i.e. 2
REWARD_SCALE = 56

"""
This class is wrapped Game environment that can be used to run multiple games
in parallet using multiprocessing. It encapsulates the agents and the game together
to make it easy for the game to played by using this class. The only thing required 
for the game is the initial parameters of the model of the agent.
"""
class RungEnv(Process):
    def __init__(self, pipe) -> None:
        super(RungEnv, self).__init__()
        self.pipe: Pipe = pipe
        self.actor = None
        self.critic = None
        self.game = None
        self.agent = PPOAgent()

    def get_params(self):
        """
        Gets the parameters of the latest model from the parent process and loads
        them into the agent
        """
        # print("COMGIN ACTOR")
        # self.actor = self.queue.get() 
        self.actor = self.pipe.recv()
        # print("COMING CITIC")
        # self.critic = self.queue.get()
        self.critic = self.pipe.recv()
        # self.actor = self.pipe.recv()
        # self.critic = self.pipe.recv()
        self.agent.load_params(self.actor, self.critic)
        # print("LOADED")
        
    def prepare_game(self):
        """
        Prepares the game by getting the latest parameters of the model from the parent
        """

        self.players = [self.agent.get_player(), 
                        self.agent.get_player(),
                        self.agent.get_player(),
                        self.agent.get_player()]
                    
        self.game = Game(self.players)
        

    def run(self):

        while True:
            msg = self.pipe.recv()
            # print(msg)
            if msg == "REFRESH":
                # print("Getting new parameters and starting a new game: ")
                self.get_params()
                # print("GOT Params")
                continue
            elif msg == "RESET":
                # print("Starting a new game: ")
                pass
            elif msg == "TERMINATE":
                print("Terminate the environment instance")
                break

            # print("Preparing")
            self.prepare_game()
            self.game.initialize()
            self.game.play_game()
            self.agent.gather_experience()
            self.send_data()

                

    def send_data(self):
        # print("Game ended")
        self.pipe.send("END")
        # self.pipe.send("STATE")
        self.pipe.send(self.agent.state_batch)
        # self.pipe.send("ACTION")
        self.pipe.send(self.agent.action_batch)
        # self.pipe.send("REWARD")
        self.pipe.send(self.agent.reward_batch)
        # self.pipe.send("LOGPROBS")
        self.pipe.send(self.agent.log_probs_batch)
        self.agent.clear_experience()
        pass
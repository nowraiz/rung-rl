# RungRL

This repo contains different RL agents for Rung. Here is an example of how to play game against AI


## Example

```python
from rung_rl.agents.dqn.dqn_agent import DQNAgent
from rung_rl.agents.human_agent import HumanAgent
from rung_rl.rung import Game
agent = DQNAgent() # create a DQN agent
agent.eval = True # run the agent in evaluation mode
agent.load_model("final") # load the last trained model 


# setup the players
players = [agent, HumanAgent(), agent, agent] # set every other player to the loaded DQN agent

# setup the game
game = Game(players)
game.initialize()
game.play_game()

# follow the inputs from the human agent to play the game. You can access state of the game at any
# stage by accessing fields from the Game object. 

# At each game step the game will invoke agent.get_move(state) to get the move from the agents
# that are specified in the game

```
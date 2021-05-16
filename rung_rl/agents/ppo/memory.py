from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'log_prob'))


"""
This class provides a replay buffer to store finite state transition tuples
as defined above to later be used for sampling and training any agent that 
requires.
"""


class Memory(object):

    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))
        # if len(self.memory) < self.capacity:
            # self.memory.append(None)
        # self.memory[self.position] = Transition(*args)
        # self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory = []
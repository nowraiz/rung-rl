import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'action_mask', 'hidden_state', 'next_hidden'))

StateAction = namedtuple('StateAction', ('state', 'action'))

MIN_PROB = 0.00

"""
This class provides a replay buffer to store finite state transition tuples
as defined above to later be used for sampling and training any agent that 
requires.
"""


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ActionMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.entries_seen = 0

    def push_with_sampling(self, *args):
        """Saves a transition with reservoir sampling"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.memory[self.position] = StateAction(*args)
            self.position += 1
        else:
            prob_add = max(float(self.capacity) / float(self.entries_seen), MIN_PROB)
            if random.random() < prob_add:
                idx = random.randint(0, self.capacity - 1)
                self.memory[idx] = StateAction(*args)
        self.entries_seen += 1

    def push(self, *args):
        raise NotImplementedError
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = StateAction(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

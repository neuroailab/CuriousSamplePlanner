import random
from collections import namedtuple
import numpy as np

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))


class BalancedReplayMemory(object):

    def __init__(self, capacity, num_classes = 2, split=0.2):
        self.capacity = capacity
        self.split = split
        self.memory_banks = [[] for _ in range(num_classes)]

    def push(self, class_index, *args):
        # print("Fail: "+str(len(self.memory_banks[0]))+" Success: "+str(len(self.memory_banks[1])))
        """Saves a transition."""
        if len(self.memory_banks[class_index]) > self.capacity:
            del self.memory_banks[class_index][0]
        self.memory_banks[class_index].append(Transition(*args))


    def sample(self, batch_size):
        returning = []
        for _ in range(batch_size):
            index = np.random.choice(np.arange(len(self.memory_banks)), p=[1-self.split, self.split])
            returning.append(random.choice(self.memory_banks[index]))
        return returning

    def __len__(self):
        return sum([len(i) for i in self.memory_banks])

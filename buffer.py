import random
import numpy as np

class ReplayBuffer(object):
    def __init__(self):
        # self.buffer = []
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        
    def add(self, s0, a, r, s1, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        # self.buffer.append((s0[None, :], a, r, s1[None, :], done))
        self.buffer.append((s0, a, r, s1, done))

    '''
    def sample(self, batch_size):
        s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        # arr1 = np.array(s0)
        return np.concatenate(s0), a, r, np.concatenate(s1), done
    '''

    def size(self):
        return len(self.buffer)
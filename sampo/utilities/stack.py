import numpy as np


# Here is the fastest stack. No checks will be performed in runtime.
class Stack(object):
    def __init__(self, max_length, dtype=np.float32):
        self.data = np.zeros(max_length, dtype=dtype)
        self.length = 0

    def push(self, x):
        self.data[self.length] = x
        self.length += 1

    def pop(self):
        self.length -= 1
        return self.data[self.length]

    def length(self):
        return self.data.size

    def swap(self, ind1: int, ind2: int):
        self.data[ind1], self.data[ind2] = self.data[ind2], self.data[ind1]

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, idx, item):
        self.data[idx] = item

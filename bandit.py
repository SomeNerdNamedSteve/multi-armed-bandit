import numpy as np

seed_val = 23682

class Bandit():

    def __init__(self, m):
        np.random.seed(seed_val)
        self.m = m
        self.N = 0
        self.mean = 0

    def pull(self): return np.random.randn() + self.m
    
    def update(self, x):
        self.N += 1
        self.mean = (1 - 1/self.N) * self.mean + 1.0/self.N*x
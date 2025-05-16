import numpy as np
from .base_scheduler import Scheduler

class LR_cosine_annealing(Scheduler):
    def __init__(self, optimizer, T, min_alfa, max_alfa):
        super().__init__(optimizer, T=T, min_alfa=min_alfa, max_alfa=max_alfa)

    def update(self):
        self.optimizer.alfa = self.min_alfa + (self.max_alfa - self.min_alfa) * (1 + np.cos(np.pi * self.t / self.T)) / 2
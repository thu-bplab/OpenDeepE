import random
import time
import datetime
import sys
import torch
import numpy as np
from torch.autograd import Variable
from visdom import Visdom

        
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_epochs, decay_factors):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_epochs = decay_epochs
        self.decay_factors = decay_factors

    def step(self, epoch):
        if epoch < self.decay_epochs[0]:
            return 1.0
        elif epoch < self.decay_epochs[1]:
            return self.decay_factors[0]
        else:
            return self.decay_factors[1]

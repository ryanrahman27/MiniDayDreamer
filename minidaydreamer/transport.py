"""Detect a pause or reversal after the arm has started carrying the lifted cube."""

from collections import deque

import numpy as np

from .stack import STEP


WINDOW = 12


class CarryStall:
    def __init__(self, first):
        self.initial_z = first[5]
        self.distances = deque(maxlen=WINDOW + 1)
        self.peak_mm = 0.
        self.transported = False

    def carrying(self, obs):
        return obs[9] > .5 and obs[10] > .5 and obs[5] - self.initial_z > .2

    def observe(self, obs):
        distance = 100 * np.linalg.norm(obs[6:8] - obs[3:5])
        if not self.carrying(obs) or distance <= 12:
            self.peak_mm = 0.
            self.transported = False
            self.distances.clear()
            return False
        self.peak_mm = max(self.peak_mm, distance)
        self.transported |= self.peak_mm - distance >= 15
        if self.transported:
            self.distances.append(distance)
        return len(self.distances) == WINDOW + 1 and self.distances[0] - distance <= 2


def targetward_xy(obs):
    # Same relative Cartesian command units as the expert. Preserve the z and grip proposal.
    delta = .1 * (obs[6:8] - obs[3:5]) / STEP
    return .85 * delta / max(1., np.linalg.norm(delta))

import collections
import numpy as np
from toybox.toys.core import Pattern


class BiCrystal(collections.MutableSequence):

    def __init__(self, pattern1, pattern2, profile=np.linspace(0, 1, 11)):
        self.pattern_1 = pattern1
        self.pattern_2 = pattern2
        self.profile = profile

    @property
    def profile(self):
        return self._profile

    @profile.setter
    def profile(self, profile):
        if np.max(profile) > 1 or np.min(profile) < 0:
            raise ValueError("Profile must be between 0 and 1.")
        self._profile = profile

    @property
    def profile_i(self):
        return 1. - self.profile

    @property
    def patterns(self):
        return np.array([p * self.pattern_2 + q * self.pattern_1 for p, q in zip(self.profile, self.profile_i)])

    def __len__(self):
        return len(self.profile)

    def __getitem__(self, item):
        return self.patterns[item].view(Pattern)

    def __setitem__(self, key, value):
        if value > 1 or value < 0:
            raise ValueError("Profile must be between 0 and 1.")
        self.profile[key] = value

    def __delitem__(self, key):
        self.profile = np.delete(self.profile, key, None)

    def insert(self, index, value):
        self.profile = np.insert(self.profile, index, value)



















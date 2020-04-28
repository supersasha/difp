import json

import numpy as np

from utils import cv65to31

class Datasheet:
    def __init__(self, filename):
        ds = None
        with open(filename, 'r') as f:
            ds = json.load(f)
        cs = ['red', 'green', 'blue']
        self.sense = np.array([cv65to31(ds[c]['sense']) for c in cs])
        self.dyes = np.array([cv65to31(ds[c]['dye']['data']) for c in cs])


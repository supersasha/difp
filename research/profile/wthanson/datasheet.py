import sys
import json

import numpy as np

from utils import cv65to31

class Datasheet:
    def __init__(self, filename):
        ds = None
        with open(filename, 'r') as f:
            ds = json.load(f)
        cv = cv65to31
        if 'samples' in ds and ds['samples'] == 31:
            cv = lambda x: x
        cs = ['red', 'green', 'blue']
        self.sense = np.array([cv(ds[c]['sense']) for c in cs], dtype=float)
        self.dyes = np.array([cv(ds[c]['dye']['data']) for c in cs], dtype=float)
    def correct(self, ymin, ycorr):
        for i in range(3):
            for j in range(31):
                if self.sense[i, j] < ymin:
                    self.sense[i, j] = ymin - ycorr
    def to_json(self):
        obj = {
            'samples': 31,
            'red': {
                'sense': self.sense[0].tolist(),
                'dye': { 'data': self.dyes[0].tolist() }
            },
            'green': {
                'sense': self.sense[1].tolist(),
                'dye': { 'data': self.dyes[1].tolist() }
            },
            'blue': {
                'sense': self.sense[2].tolist(),
                'dye': { 'data': self.dyes[2].tolist() }
            }
        }
        return json.dumps(obj, indent=4)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage', file=sys.stderr)
        sys.exit(1)
    filename = sys.argv[1]
    ymin = float(sys.argv[2])
    ycorr = float(sys.argv[3])
    ds = Datasheet(sys.argv[1])
    ds.correct(ymin, ycorr)
    print(ds.to_json())

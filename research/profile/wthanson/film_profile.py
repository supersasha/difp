import json
import numpy as np
from scipy.interpolate import CubicSpline

import illum
import utils

class FilmProfile:
    def __init__(self, p = None, mode31=False):
        self.mode31 = mode31
        if type(p) is dict:
            self.set(p)
        elif type(p) is str:
            self.load(p)

    def load(self, filename):
        with open(filename, 'r') as fp:
            profile = json.load(fp)
        self.set(profile)

    def set(self, profile):
        self.profile = profile
        self.curves = {}
        for k in ['red', 'green', 'blue']:
            nodes = self.profile[k]['curve']['nodes']
            x = np.array([p['x'] for p in nodes])
            y = np.array([p['y'] for p in nodes])
            if len(x) == len(y) and len(x) > 1:
                self.curves[k] = CubicSpline(x, y, bc_type='natural',
                                                extrapolate=True)

    def key_by_index(self, idx):
        if idx == 0:
            return 'red'
        elif idx == 1:
            return 'green'
        elif idx == 2:
            return 'blue'
        return None

    def dye(self, idx = None):
        if idx is None:
            return np.vstack([
                self._m31(np.array(self.profile['red'  ]['dye']['data'])),
                self._m31(np.array(self.profile['green']['dye']['data'])),
                self._m31(np.array(self.profile['blue' ]['dye']['data']))
            ])
        key = self.key_by_index(idx)
        return self._m31(np.array(self.profile[key]['dye']['data']))

    def sense(self, idx = None):
        if idx is None:
            return np.vstack([
                self._m31(np.array(self.profile['red'  ]['sense'])),
                self._m31(np.array(self.profile['green']['sense'])),
                self._m31(np.array(self.profile['blue' ]['sense']))
            ])
        key = self.key_by_index(idx)
        return self._m31(np.array(self.profile[key]['sense']))

    def curve(self, idx, x=None):
        key = self.key_by_index(idx)
        x0, x1 = self.curve_arg_range(idx)
        if x is None:
            return self.curves[key](np.linspace(x0, x1, 100))
        else:
            return self.curves[key](np.clip(x, x0, x1))

    def curve_arg_range(self, idx):
        key = self.key_by_index(idx)
        nodes = self.profile[key]['curve']['nodes']
        return (nodes[0]['x'], nodes[-1]['x'])
    
    def clone(self):
        p = FilmProfile()
        p.set(json.loads(json.dumps(self.profile)))
        return p

    def __str__(self):
        return json.dumps(self.profile, indent=4)

    def _m31(self, v):
        if self.mode31:
            return utils.to_400_700_10nm(v)
        return v

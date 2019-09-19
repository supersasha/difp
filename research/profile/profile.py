import json
import numpy as np
from scipy.interpolate import CubicSpline

import illum
from rgb import Rgb

class FilmProfile:
    def __init__(self, p = None):
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
            return np.vstack((np.array(self.profile['red']['dye']['data']),
                             np.array(self.profile['green']['dye']['data']),
                             np.array(self.profile['blue']['dye']['data'])))
        key = self.key_by_index(idx)
        return np.array(self.profile[key]['dye']['data'])

    def sense(self, idx):
        key = self.key_by_index(idx)
        return np.array(self.profile[key]['sense'])

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

    def expose(self, ill0, h):
        tr = self.profile["red"]["theta"]
        tg = self.profile["green"]["theta"]
        tb = self.profile["blue"]["theta"]
        ill = 10**h * illum.normalize(ill0)
        r = self.curve(0, np.log10(ill @ 10 ** (self.sense(0)))-tr)#
        g = self.curve(1, np.log10(ill @ 10 ** (self.sense(1)))-tg)#
        b = self.curve(2, np.log10(ill @ 10 ** (self.sense(2)))-tb)#
        return Rgb(r, g, b)

    def d0(self, idx):
        key = self.key_by_index(idx)
        return self.profile[key]['d0']

    def __str__(self):
        return json.dumps(self.profile, indent=4)

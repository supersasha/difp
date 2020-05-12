import math
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

import illum
import brewer
from datasheet import Datasheet
from utils import vectorize
import spectrum
import colors

import nlopt

def bell(q, x):
    # q=[a, mu, sigma]
    return q[0]*np.exp(-((x-q[1])/q[2])**2)

def chi_f(xmin, xmax, ymin, ymax):
    @vectorize()
    def chi(x):
        if x < xmin:
            return ymin
        if x > xmax:
            return ymax
        return (x - xmin) / (xmax - xmin) * (ymax - ymin) + ymin
    return chi

def qdye(idx, logsense, light, refl, chi):
    return chi(np.log10((10 ** logsense[idx]) @ (light * refl)))

def reference_colors():
    def srgb_to_xyz(r, g, b):
        return colors.srgb_to_xyz(colors.color(r, g, b))
    vs = np.linspace(0.1, 1, 5);
    reds =   [(srgb_to_xyz(v, 0, 0), 1) for v in vs]
    greens = [(srgb_to_xyz(0, v, 0), 1) for v in vs]
    blues =  [(srgb_to_xyz(0, 0, v), 1) for v in vs]
    grays =  [(srgb_to_xyz(v, v, v), 1) for v in vs]
    xyzs = reds + greens + blues + grays
    return xyzs

class Couplers:
    def __init__(self):
        self.film = Datasheet(
            '../../../profiles/datasheets/kodak-vision3-250d-5207.datasheet')
        self.paper = Datasheet(
            '../../../profiles/datasheets/kodak-vision-color-print-2383.datasheet')
        ixs = list(range(3))
        self.sense_film = brewer.normalized_sense(self.film.sense, illum.D55_31)
        self.kfmax = 2.5
        self.chi_film = chi_f(-self.kfmax, 0.0, 0.0, self.kfmax) # gamma = 1

        self.sense_paper = brewer.normalized_sense(self.paper.sense, illum.D55_31)
        self.T1 = 1.0 / 10 ** np.sum(self.film.dyes, axis=0)
        Tw = self.T1 ** self.kfmax
        self.beta = [
            math.log10(10**self.sense_paper[i] @ (illum.D55_31 * Tw))
            for i in ixs
        ]

        self.kpmax = 4.0
        self.chi_paper = [ chi_f(self.beta[i], 0, 0, self.kpmax) for i in ixs ]


        self.refl_gen = spectrum.load_spectrum('spectra2/spectrum-d55-4.json')
        self.mtx_refl = spectrum.transmittance_to_xyz_mtx(illum.D65_31)
    
    def make_couplers(self, q):
        couplers = []
        xs = np.arange(31)*10.0 + 400
        for idx in range(3):
            b = 9 + idx*6
            couplers.append(bell(q[b:b+3], xs) + bell(q[b+3:b+6], xs))
        self.couplers = np.vstack(couplers)

    def to_json(self):
        obj = {
            'film_sense': self.sense_film.tolist(),
            'film_dyes': self.film.dyes.tolist(),
            'paper_sense': self.sense_paper.tolist(),
            'paper_dyes': self.paper.dyes.tolist(),
            'couplers': self.couplers.tolist(), #np.zeros((3, 31), dtype=float).tolist(),
            'proj_light': illum.D55_31.tolist(),
            'dev_light': illum.D55_31.tolist(),
            'mtx_refl': self.mtx_refl.tolist(),
            'neg_gammas': [1, 1, 1],
            'paper_gammas': [-self.kpmax/self.beta[i] for i in range(3)],
            'film_max_qs': [1, 1, 1], # this is useless for now
        }
        return json.dumps(obj, indent=4)

    def solve(self):
        xyzs = reference_colors()

        def f(x):
            self.make_couplers(x)
            d = 0
            for xyz, v in xyzs:
                xyz1 = self.develop(xyz, x)
                d0 = colors.delta_E76_xyz(xyz, xyz1) * v
                d += d0*d0
            return d
        
        gmin = 1.0e10
        step = 0
        def f_nl(x, grad):
            nonlocal step
            nonlocal gmin
            step += 1
            g = f(x)
            if g < gmin:
                gmin = g
                print(step, g, x, file=sys.stderr)
            return g

        bounds = [(0.01, 1)] * 3 + [(1, 5)] * 3 + [(-2, 2)] * 3 +[(0.0, 1), (350.0, 750.0), (20.0, 200.0)] * 6
         
        def cb(x, f, ctx):
            print(f, x.reshape((10, 3)), ctx, file=sys.stderr)

        # GN_CRS2_LM
        optm = nlopt.opt(nlopt.GN_ISRES, len(bounds))
        optm.set_min_objective(f_nl)
        lb = np.array(list(list(zip(*bounds))[0]))
        ub = np.array(list(list(zip(*bounds))[1]))
        optm.set_lower_bounds(lb)
        optm.set_upper_bounds(ub)
        optm.set_maxtime(60*30) #(3600 * 3)
        #xopt = optm.optimize(lb) #((lb + ub)/2)
        xopt = optm.optimize([ 1, 1, 1, 1.6598, 1.5639, 1.6372, 0, 0, 0 ] + [0, 350, 20] * 6)
        self.solution = xopt

    def develop(self, xyz, q):
        ixs = list(range(3))
        self.chi_film = [ chi_f(-self.kfmax/q[i], 0, 0, self.kfmax) for i in ixs ]
        self.chi_paper = [ chi_f(-self.kpmax/q[3+i], 0, 0, self.kpmax) for i in ixs ]
        sp, refl = self.refl_gen.spectrum_of(xyz)
        kf = np.array([
            qdye(i, self.sense_film, illum.D55_31, refl, self.chi_film[i])
            for i in ixs
        ])
        #print('kf:', kf, file=sys.stderr)
        T = 1.0 / 10**(kf @ self.film.dyes + (1 - kf/self.kfmax) @ self.couplers)
        kp = np.array([
            qdye(i, (self.sense_paper.transpose() + q[6:9]).transpose(), illum.D55_31, T, self.chi_paper[i])
            for i in ixs
        ])
        #print('kp:', kp, file=sys.stderr)
        trans = brewer.transmittance(self.paper.dyes, kp)
        xyz1 = spectrum.transmittance_to_xyz(self.mtx_refl, trans)
        #print(xyz1, '==', colors.xyz_to_srgb(xyz1), file=sys.stderr)
        return xyz1

def main():
    nc = Couplers()
    nc.solve()
    print(nc.to_json())

if __name__ == '__main__':
    main()


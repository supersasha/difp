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

def chi_f(xmin, xmax, ymin, ymax):
    @vectorize()
    def chi(x):
        if x < xmin:
            return ymin
        if x > xmax:
            return ymax
        return (x - xmin) / (xmax - xmin) * (ymax - ymin) + ymin
    return chi

def chi_to_f(ymin, ymax, gamma, xmax):
    xmin = xmax - (ymax - ymin) / gamma
    return chi_f(xmin, xmax, ymin, ymax)

def chi_from_f(ymin, ymax, gamma, xmin):
    xmax = xmin + (ymax - ymin) / gamma
    return chi_f(xmin, xmax, ymin, ymax)

def qdye(idx, logsense, light, refl, chi):
    return chi(np.log10((10 ** logsense[idx]) @ (light * refl)))

class NoCouplers:
    def __init__(self):
        self.film = Datasheet(
            #'../../../profiles/datasheets/kodak-vision3-250d-5207.datasheet')
            '../../../profiles/datasheets/square.datasheet')
        self.paper = Datasheet(
            #'../../../profiles/datasheets/kodak-vision-color-print-2383.datasheet')
            '../../../profiles/datasheets/square.datasheet')
        ixs = list(range(3))
        self.sense_film = brewer.normalized_sense(self.film.sense, illum.D55_31)
        self.kfmax = 2.5
        #self.chi_film = chi_f(-self.kfmax-3.0, 0.0, 0.0, self.kfmax) # gamma = 1
        self.chi_film = chi_to_f(0.0, self.kfmax, 0.4545, 0) # gamma = 1

        self.sense_paper = brewer.normalized_sense(self.paper.sense, illum.D55_31)
        #self.T1 = 1.0 / 10 ** np.sum(self.film.dyes, axis=0)
        self.T1 = 10 ** (-np.ones(3, dtype=float) @ self.film.dyes)
        Tw = self.T1 ** self.kfmax
        self.beta = [
            math.log10(10**self.sense_paper[i] @ (illum.D55_31 * Tw))
            for i in ixs
        ]

        print('beta:', self.beta)

        self.kpmax = 4.0
        #self.chi_paper = [ chi_f(self.beta[i], 0, 0, self.kpmax) for i in ixs ]
        self.chi_paper = [ chi_to_f(0, self.kpmax, -self.kpmax/self.beta[i], 0) for i in ixs ]

        self.refl_gen = spectrum.load_spectrum('spectra2/spectrum-d55-4.json')
        self.mtx_refl = spectrum.transmittance_to_xyz_mtx(illum.D65_31)

    def to_json(self):
        obj = {
            'film_sense': self.sense_film.tolist(),
            'film_dyes': self.film.dyes.tolist(),
            'paper_sense': self.sense_paper.tolist(),
            'paper_dyes': self.paper.dyes.tolist(),
            'couplers': np.zeros((3, 31), dtype=float).tolist(),
            'proj_light': illum.D55_31.tolist(),
            'dev_light': illum.D55_31.tolist(),
            'mtx_refl': self.mtx_refl.tolist(),
            'neg_gammas': [0.4545, 0.4545, 0.4545],
            'paper_gammas': [-self.kpmax/self.beta[i] for i in range(3)],
            'film_max_qs': [1, 1, 1], # this is useless for now
        }
        return json.dumps(obj, indent=4)

    def work(self, xyz):
        sp, refl = self.refl_gen.spectrum_of(xyz)
        ixs = list(range(3))
        kf = np.array([
            qdye(i, self.sense_film, illum.D55_31, refl, self.chi_film)
            for i in ixs
        ])
        T = 10**(-kf @ self.film.dyes)
        kp = np.array([
            qdye(i, self.sense_paper, illum.D55_31, T, self.chi_paper[i])
            for i in ixs
        ])
        trans = 10**(-kp @ self.paper.dyes)
        xyz1 = spectrum.transmittance_to_xyz(self.mtx_refl, trans)
        return xyz1

def main():
    nc = NoCouplers()
    xs = np.linspace(0, 1, 1000)
    cs = [[], [], []]
    cs0 = [[], [], []]
    for x in xs:
        rgb = colors.color(x, x, x)
        xyz = colors.srgb_to_xyz(rgb)
        xyz1 = nc.work(x*xyz)
        rgb1 = colors.xyz_to_srgb(xyz1)
        cs[0].append(xyz1[0])
        cs[1].append(xyz1[1])
        cs[2].append(xyz1[2])
        cs0[0].append(xyz[0])
        cs0[1].append(xyz[1])
        cs0[2].append(xyz[2])
    cs = np.array(cs)
    cs0 = np.array(cs0)
    plt.plot(xs, cs[0], 'r')
    plt.plot(xs, cs[1], 'g')
    plt.plot(xs, cs[2], 'b')
    plt.plot(xs, cs0[0], 'c')
    plt.plot(xs, cs0[1], 'm')
    plt.plot(xs, cs0[2], 'y')
    plt.show()
    #print(nc.to_json())


if __name__ == '__main__':
    main()


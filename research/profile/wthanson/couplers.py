import math
import random
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

def ones_at(f, t):
    xs = np.zeros(31, dtype=float)
    for i in range(31):
        if i >= f and i < t:
            xs[i] = 1.0
    return xs

def bell(a, mu, sigma):
    x = np.arange(31, dtype=float)
    x = x*10 + 400
    return a*np.exp(-((x-mu)/sigma)**2)

class Couplers:
    def __init__(self):
        self.reset()

    def reset(self):
        self.couplers = np.zeros((3, 31), dtype=float)

    def add_bell(self, layer, a, mu, sigma):
        self.couplers[layer] += bell(a, mu, sigma)

    def result(self):
        return self.couplers

class Profile:
    def __init__(self):
        self.film = Datasheet(
            '../../../profiles/datasheets/kodak-vision3-250d-5207-2.datasheet')
            #'../../../profiles/datasheets/square.datasheet')
        self.paper = Datasheet(
            '../../../profiles/datasheets/kodak-vision-color-print-2383-2.datasheet')
            #'../../../profiles/datasheets/square.datasheet')
        self.sense_film = brewer.normalized_sense(self.film.sense, illum.D55_31)
        self.dyes_film = brewer.normalized_dyes(self.film.dyes, illum.D55_31, 1.0)
        self.sense_paper = brewer.normalized_sense(self.paper.sense, illum.D55_31)
        self.dyes_paper0 = brewer.normalized_dyes(self.paper.dyes, illum.D65_31, 1.0)
        self.refl_gen = spectrum.load_spectrum('spectra2/spectrum-d55-4.json')
        self.mtx_refl = spectrum.transmittance_to_xyz_mtx(illum.D65_31)

    def set_couplers(self, couplers):

        ixs = list(range(3))
        self.couplers = couplers

        #print(self.couplers, file=sys.stderr)

        self.kfmax = 2.5
        self.film_gammas = np.array([0.5, 0.5, 0.5])
        self.chi_film = [
            chi_to_f(0.0, self.kfmax, self.film_gammas[i], 0)
            for i in ixs
        ]

        self.kpmax = 4.0
        self.hproj_max = self.beta(-10.0)
        self.hproj_min = self.beta(1.0)
        self.paper_gammas = self.kpmax / (self.hproj_max - self.hproj_min)
        self.chi_paper = [
            chi_to_f(0, self.kpmax, self.paper_gammas[i], self.hproj_max[i])
            for i in ixs
        ]
        self.dyes_paper = self.dyes_paper0 * 4 / self.delta(-4) # this is not exact!

    
    def beta(self, D):
        alpha = D
        kfs = np.array([
            self.chi_film[0](alpha),
            self.chi_film[1](alpha),
            self.chi_film[2](alpha),
        ])
        trans = 10 ** -(kfs @ self.dyes_film + (1 - kfs/self.kfmax) @ self.couplers)
        return np.log10(10**self.sense_paper @ (illum.D55_31 * trans))

    def beta_v(self, Ds):
        return np.vstack([self.beta(d) for d in Ds]).transpose()

    def delta(self, D):
        betas = self.beta(D)
        kps = np.array([
            self.chi_paper[0](betas[0]),
            self.chi_paper[1](betas[1]),
            self.chi_paper[2](betas[2]),
        ])
        refl = 10 ** -(kps @ self.dyes_paper0)
        return np.log10((illum.D65_31 @ np.ones(31, dtype=float)) / (illum.D65_31 @ refl))
    
    def delta_v(self, Ds):
        return np.vstack([self.delta(d) for d in Ds]).transpose()

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

    def to_json(self):
        obj = {
            'film_sense': self.sense_film.tolist(),
            'film_dyes': self.dyes_film.tolist(),
            'paper_sense': self.sense_paper.tolist(),
            'paper_dyes': self.dyes_paper.tolist(),
            'couplers': self.couplers.tolist(), #np.zeros((3, 31), dtype=float).tolist(),
            'proj_light': illum.D55_31.tolist(),
            'dev_light': illum.D55_31.tolist(),
            'mtx_refl': self.mtx_refl.tolist(),
            'neg_gammas': self.film_gammas.tolist(),
            'paper_gammas': self.paper_gammas.tolist(),
            'film_max_qs': self.hproj_max.tolist(),
            #[1, 1, 1], # this is useless for now
        }
        return json.dumps(obj, indent=4)

def main():
    random.seed()
    def rnd():
        return random.random() * 1.5

    def rnd_layer():
        return ones_at(0, 10) * rnd() + ones_at(10, 20) * rnd() + ones_at(20, 31) * rnd()

    couplers = np.zeros((3, 31), dtype=float)
    couplers[0] = rnd_layer() 
    couplers[1] = rnd_layer() 
    couplers[2] = rnd_layer() 

    nc = Profile()
    nc.set_couplers(couplers)
    ds = np.linspace(-10, 4, 1000)
    ys = nc.delta_v(ds)
    #print(ys)
    plt.plot(ds, ys[0], 'r')
    #plt.plot(ds, ys[1], 'g')
    #plt.plot(ds, ys[2], 'b')
    plt.grid()


    print(nc.to_json())
    plt.show()



if __name__ == '__main__':
    main()


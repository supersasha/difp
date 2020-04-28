import math

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

def qdye(idx, logsense, light, refl, chi):
    return chi(np.log10((10 ** logsense[idx]) @ (light * refl)))

class NoCouplers:
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

    def work(self, xyz):
        sp, refl = self.refl_gen.spectrum_of(xyz)
        ixs = list(range(3))
        kf = np.array([
            qdye(i, self.sense_film, illum.D55_31, refl, self.chi_film)
            for i in ixs
        ])
        print('kf:', kf)
        T = 1.0 / 10**(kf @ self.film.dyes)
            # == (kf[0] * film.dyes[0] + kf[1] * film.dyes[1] + kf[2] * film.dyes[2])
        kp = np.array([
            qdye(i, self.sense_paper, illum.D55_31, T, self.chi_paper[i])
            for i in ixs
        ])
        print('kp:', kp)
        trans = brewer.transmittance(self.paper.dyes, kp)
        xyz1 = spectrum.transmittance_to_xyz(self.mtx_refl, trans)
        print(xyz1, '==', colors.xyz_to_srgb(xyz1))

def main():
    nc = NoCouplers()
    nc.work(colors.srgb_to_xyz(colors.color(0.5, 0.5, 0.5)))

if __name__ == '__main__':
    main()


import numpy as np
import math
import colors
import spectrum
import utils
import illum
import spectral_bases
import data

import matplotlib.pyplot as plt
import random
from matplotlib.patches import Rectangle

def spectrum_to_xyz_mtx(ill):
    N = np.dot(data.A_1931_64_400_700_10nm.transpose()[1], ill)
    mtx = data.A_1931_64_400_700_10nm.transpose() * (100 / N * ill)
    return mtx

def spectrum_to_xyz(mtx, sp):
    return np.dot(mtx, sp)

class Bases:
    def __init__(self, lams, light):
        self.light = light
        self.wp = colors.chromaticity(
                np.dot(data.A_1931_64_400_700_10nm.transpose(), self.light)
        )
        print('WP:', self.wp)
            #np.array([1/3, 1/3]) # white point aka center
        self.sectors = np.array([
            colors.spectral_color_xy(lam) - self.wp for lam in lams
        ])
        self.mtx = spectrum_to_xyz_mtx(self.light)
        self.bases = []

    def extract(self):
        d = spectral_bases.load_spectra()
        t = spectrum_to_xyz(self.mtx, d.transpose()).transpose()
        #t = data.A_1931_64_400_700_10nm.transpose().dot(d.transpose()).transpose()
        tv = t[:,0] + t[:, 1] + t[:, 2]
        print(f'Y: ({np.min(t[:,1])}, {np.max(t[:,1])})')
        xy = (t.transpose() / tv).transpose()[:, 0:2]

        sector_samples = []
        for i in range(len(self.sectors)+1):
            sector_samples.append([])

        for i in range(d.shape[0]):
            s = self.find_sector(xy[i])
            sector_samples[s].append(d[i])

        for i in range(len(sector_samples)):
            samples = np.vstack(sector_samples[i])
            print(f'Sector #{i}: {len(samples)} samples')
            b = spectral_bases.extract_basis(samples)
            if i == -1:
                plt.plot(b)
                plt.show()
            self.bases.append(b)

    def check(self, n, sector=None):
        levels = np.linspace(0.0, 1.0, n)
        rs = [spectrum.Reflectance(self.light, b) for b in self.bases]
        d = 0
        for r in levels:
            for g in levels:
                for b in levels:
                    rgb = colors.color(r, g, b)
                    xyz = colors.srgb_to_xyz(rgb)
                    xy = colors.chromaticity(xyz)
                    #print(xy)
                    ns = self.find_sector(xy)
                    if sector is not None and ns != sector:
                        continue
                    rr = rs[ns]
                    refl = rr.from_xyz(xyz)
                    sp = refl * self.light
                    xyz1 = spectrum_to_xyz(self.mtx, refl.clip(0, None))
                    dist = colors.delta_E76_xyz(xyz, xyz1)
                    if dist > 5: 
                        print(dist, xyz, xyz1)
                    plt.plot(sp)
                    sp = np.clip(sp, None, 0)
                    d += np.sum(sp*sp)
        print(d)
        plt.show()

    def xyz_to_spectrum(self, xyz):
        xy = colors.chromaticity(xyz)
        s = self.find_sector(xy)
        r = spectrum.Reflectance(self.light, self.bases[s])
        refl = r.from_xyz(xyz)
        #if refl.max() > 1:
        #    refl = refl / refl.max()
        refl = refl.clip(0, 1)
        sp = refl * self.light
        xyz1 = spectrum_to_xyz(self.mtx, refl)
        def rect(x, y, w, h, xyz):
            plt.gca().add_patch(
                Rectangle((x, y), w, h, color=colors.xyz_to_srgb(xyz)))
        rect(400, 0, 150, 1, xyz)
        rect(550, 0, 150, 1, xyz1)
        print(f'{colors.delta_E76_xyz(xyz, xyz1):4.1f}', colors.xyz_to_srgb(xyz))
        plt.plot(np.linspace(400, 700, 31), refl)
        #plt.plot(np.linspace(400, 700, 31), sp)
        plt.show()

    def find_sector(self, xy):
        '''
            Be sure to always use reflected chromaticity
            So we need to use light source
        '''
        xy1 = xy - self.wp
        for i in range(len(self.sectors) - 1):
            if self.in_sector(i, xy1):
                if i == 0 and math.sqrt(np.sum(xy1*xy1)) < 0.16:
                    return len(self.sectors)
                return i
        return len(self.sectors) - 1

    def in_sector(self, sec_num, c):
        v1 = self.sectors[sec_num]
        v2 = self.sectors[sec_num+1]
        d = v1[0]*v2[1] - v2[0]*v1[1]
        q1 = (c[0]*v2[1] - c[1]*v2[0]) / d
        q2 = (-c[0]*v1[1] + c[1]*v1[0]) / d
        return q1 > 0 and q2 > 0

if __name__ == '__main__':
    light = utils.to_400_700_10nm(illum.D55)
    b = Bases([380, 475, 490, 560, 580, 700], light) #605
    b.extract()
    #for i in range(len(b.bases)):
    #    b.check(10, sector=i)
    #srgb = colors.color(0.883, 0.831, 0.775)

    while True:
        srgb = colors.color(random.random(), random.random(), random.random())
        #print(srgb)
        xyz = colors.srgb_to_xyz(srgb)
        b.xyz_to_spectrum(xyz)

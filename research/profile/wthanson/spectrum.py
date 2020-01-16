import math
import json

import numpy as np
from numpy.linalg import inv
from scipy.io import loadmat, savemat
from sklearn.decomposition import NMF

import data
import colors
import utils

# Based on Amirshahi 2010
# with additional chromaticity sector clustering

def extract_basis(spectra):
    model = NMF(n_components=3, max_iter=1000, tol=1e-20)
    U = model.fit_transform(spectra.transpose())
    return U

def load_spectra():
    mat = loadmat('TotalRefs_IndividualSpectra.mat')
    return mat['TotalRefs_IndividualSpectra']

def reflectance_to_xyz_mtx(light):
    N = np.dot(data.A_1931_64_400_700_10nm.transpose()[1], light)
    mtx = data.A_1931_64_400_700_10nm.transpose() * (100 / N * light)
    return mtx

def reflectance_to_xyz(mtx, refl):
    return np.dot(mtx, refl)

transmittance_to_xyz_mtx = reflectance_to_xyz_mtx
transmittance_to_xyz = reflectance_to_xyz

class SpectralBasis:
    def __init__(self, illuminant, basis):
        self.basis = basis
        K = 100.0
        N = illuminant.dot(data.A_1931_64_400_700_10nm[:, 1])
        self.A_T = data.A_1931_64_400_700_10nm.transpose() * illuminant * K / N
        self.tri_to_v_mtx = inv(np.dot(self.A_T, basis))
    def reflectance_of(self, xyz):
        v = np.dot(self.tri_to_v_mtx, xyz)
        return np.dot(self.basis, v)

class Spectrum:
    def __init__(self, light_or_filename, lams=list([380, 475, 490, 560, 580, 700])):
        if isinstance(light_or_filename, str):
            self.loadmat(light_or_filename)
        else:
            self.lambdas = lams
            self.light = light_or_filename
            self._do_init()
            self.bases = []

    def _do_init(self):
        self.wp = colors.chromaticity(
            data.A_1931_64_400_700_10nm.transpose() @ self.light
        )
        self.sectors = np.array([
            colors.spectral_color_xy(lam) - self.wp for lam in self.lambdas
        ])
        self.mtx = reflectance_to_xyz_mtx(self.light)

    def savemat(self, filename):
        print('saved light:', self.light)
        mdict = {
            'light': self.light,
            'lambdas': self.lambdas,
            'bases': np.array([b.basis for b in self.bases])
        }
        savemat(filename, mdict)

    def loadmat(self, filename):
        mdict = loadmat(filename)
        self.light = mdict['light'][0]
        self.lambdas = mdict['lambdas'][0]
        self.bases = [SpectralBasis(self.light, b) for b in mdict['bases']]
        self._do_init()

    def to_json(self):
        obj = {
            'wp': self.wp.tolist(), # 2
            'sectors': self.sectors.tolist(), # 6x2
            'light': self.light.tolist(), # 31
            # 7x3x31
            'bases': np.array([b.basis.transpose() for b in self.bases]).tolist(),
            # 7x3x3 
            'tri_to_v_mtx': np.array([b.tri_to_v_mtx for b in self.bases]).tolist()
        }
        return json.dumps(obj, indent=4)

    def extract_bases(self):
        d = load_spectra()
        t = reflectance_to_xyz(self.mtx, d.transpose()).transpose()
        tv = t[:,0] + t[:, 1] + t[:, 2]
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
            b = extract_basis(samples)
            self.bases.append(SpectralBasis(self.light, b))

    def spectrum_of(self, xyz):
        xy = colors.chromaticity(xyz)
        s = self.find_sector(xy)
        refl = self.bases[s].reflectance_of(xyz)
        refl = refl.clip(1.0e-15, 1)
        sp = refl * self.light
        return sp, refl

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
    import random
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import illum

    def rect(x, y, w, h, xyz):
        plt.gca().add_patch(
            Rectangle((x, y), w, h, color=colors.xyz_to_srgb(xyz)))
    
    light = utils.to_400_700_10nm(illum.D55)
    s = Spectrum(light)
    s.extract_bases()
    mtx = reflectance_to_xyz_mtx(s.light)
    while True:
        srgb = colors.color(random.random()*0.1+0.9, random.random()*0.1, random.random()*0.1)
        xyz = colors.srgb_to_xyz(srgb)
        sp, refl = s.spectrum_of(xyz)
        xyz1 = reflectance_to_xyz(mtx, refl)
        print(f'{colors.delta_E76_xyz(xyz, xyz1):4.1f}', colors.xyz_to_srgb(xyz))
        
        plt.figure(figsize=(21,7))
        plt.subplot(121)
        rect(400, 0, 150, 1, xyz)
        rect(550, 0, 150, 1, xyz1)
        plt.plot(np.linspace(400, 700, 31), refl)
        plt.subplot(122)
        rect(400, 0, 300, 100, xyz1)
        plt.plot(np.linspace(400, 700, 31), sp)
        plt.show()


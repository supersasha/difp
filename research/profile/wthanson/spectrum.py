import sys
import math
import json

import numpy as np
from numpy.linalg import inv
import numpy.random as nprnd
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

def load_spectrum(filename):
    sp = None
    with open(filename) as f:
        sp = Spectrum(f.read())
    return sp

class Spectrum:
    def __init__(self, light_or_json):
        if isinstance(light_or_json, str):
            self.from_json(light_or_json)
        else:
            self.light = light_or_json
            self._do_init()
            self.base = None
            self.spectra = load_spectra()

    def _do_init(self):
        self.wp = colors.chromaticity(
            data.A_1931_64_400_700_10nm.transpose() @ self.light
        )
        self.mtx = reflectance_to_xyz_mtx(self.light)

    def to_json(self):
        obj = {
            'wp': self.wp.tolist(), # 2
            'light': self.light.tolist(), # 31
            # 3x31
            'base': self.base.basis.transpose().tolist(),
            # 3x3 
            'tri_to_v_mtx': self.base.tri_to_v_mtx.tolist()
        }
        return json.dumps(obj, indent=4)

    def from_json(self, j):
        obj = json.loads(j)
        self.wp = np.array(obj['wp'])
        self.light = np.array(obj['light'])
        self.base = SpectralBasis(self.light, np.array(obj['base']).transpose())

    def extract_bases(self):
        d = self.spectra
        t = reflectance_to_xyz(self.mtx, d.transpose()).transpose()
        tv = t[:,0] + t[:, 1] + t[:, 2]
        xy = (t.transpose() / tv).transpose()[:, 0:2]
        
        ss = {}
        K = 10
        d1 = []
        for k in range(len(xy)):
            i = math.floor(xy[k][0] * K)
            j = math.floor(xy[k][1] * K)
            if (i, j) not in ss:
                ss[i, j] = list()
            ss[i, j].append(d[k])
        
        for s in ss.values():
            if len(s) > 0:
                d1.append(s[nprnd.choice(len(s))])
        d1 = np.array(d1)

        #d = nprnd.permutation(self.spectra)[:100]
        
        b = extract_basis(d1)
        self.base = SpectralBasis(self.light, b)
        return d1
    
    def unclipped_spectrum_of(self, xyz):
        refl = self.base.reflectance_of(xyz)
        sp = refl * self.light
        return sp, refl

    def spectrum_of(self, xyz):
        refl = self.base.reflectance_of(xyz)
        refl = refl.clip(1.0e-15, 1)
        sp = refl * self.light
        return sp, refl

if __name__ == '__main__':
    import random
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import illum

    def rect(x, y, w, h, xyz):
        plt.gca().add_patch(
            Rectangle((x, y), w, h, color=colors.xyz_to_srgb(xyz)))
    
    light = utils.to_400_700_10nm(illum.daylight(5500))
    s = Spectrum(light)
    s.extract_bases()
    #print(s.to_json())
    mtx = reflectance_to_xyz_mtx(s.light)

    def play():
        while True:
            srgb = colors.color(random.random(), random.random(), random.random())
            xyz = colors.srgb_to_xyz(srgb)
            sp, refl = s.spectrum_of(xyz)
            xyz1 = reflectance_to_xyz(mtx, refl)
            print(f'{colors.delta_E76_xyz(xyz, xyz1):4.1f}', colors.xyz_to_srgb(xyz), file=sys.stderr)
            
            plt.figure(figsize=(21,7))
            plt.subplot(121)
            rect(400, 0, 150, 1, xyz)
            rect(550, 0, 150, 1, xyz1)
            plt.plot(np.linspace(400, 700, 31), refl)
            plt.subplot(122)
            rect(400, 0, 300, 100, xyz1)
            plt.plot(np.linspace(400, 700, 31), sp)
            plt.show()

    def check():
        xyzs = []
        rng = np.linspace(0, 1, 6)
        for r in rng:
            for g in rng:
                for b in rng:
                    xyzs.append(colors.srgb_to_xyz(colors.color(r, g, b))/3)
        def check1(refls):
            err = 0
            maxerr = 0
            cmax = None
            c1max = None
            for c in xyzs:
                sp, refl = s.spectrum_of(c)
                c1 = reflectance_to_xyz(mtx, refl)
                d = colors.delta_E76_xyz(c, c1)
                err += d
                if maxerr < d:
                    maxerr = d
                    cmax = c
                    c1max = c1
            avgerr = err / len(xyzs)

            if avgerr < 2:
                print(f'Avg err: {avgerr}, Max err: {maxerr}, Max err for: {cmax} vs {c1max}', file=sys.stderr)
            if (avgerr < 1 and maxerr < 9):
                print(s.to_json(), flush=True)
                sps = refls * light
                ts = sps @ data.A_1931_64_400_700_10nm
                tv = ts @ np.ones(3)
                xy = (ts.transpose() / tv).transpose()[:, 0:2]
                Y = 25
                cs = []
                for i in range(xy.shape[0]):
                    x = xy[i, 0]
                    y = xy[i, 1]
                    xyz = colors.color(x*Y/y, Y, (1-x-y)*Y/y)
                    srgb = colors.xyz_to_srgb(xyz)
                    cs.append(srgb)
            
                plt.scatter(xy.transpose()[0], xy.transpose()[1], c=cs)
                plt.plot([colors.spectral_color_xy(l)[0] for l in np.linspace(380, 700, 65)],
                    [colors.spectral_color_xy(l)[1] for l in np.linspace(380, 700, 65)], 'c')
                plt.show()
                play()

        while True:
            refls = s.extract_bases()
            check1(refls)

    #play()
    check()


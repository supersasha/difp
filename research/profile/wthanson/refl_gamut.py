import numpy as np
import numpy.random as nprnd
import matplotlib.pyplot as plt

import spectrum
import illum
import colors
import utils
import data

def refl_gamut(spectra, light):
    sps = spectra * light
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


if __name__ == '__main__':
    mat = spectrum.load_spectra()
    light = illum.D55
    light = utils.to_400_700_10nm(light)
    refl_gamut(mat, light)

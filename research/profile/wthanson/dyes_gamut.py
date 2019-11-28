import sys
import random

import numpy as np
import numpy.random as nprnd
import matplotlib.pyplot as plt

import profile
import utils
import illum
import data
import colors
import spectrum

def dyes_gamut(dyes, light, qs=None):
    trans_to_xyz_mtx = spectrum.transmittance_to_xyz_mtx(light)
    if qs is None:
        qs = nprnd.rand(3, 100000) * 3
    trans = 1.0 / (10**(dyes.transpose() @ qs)).transpose() # 1000x31
    ts = trans @ trans_to_xyz_mtx.transpose() #data.A_1931_64_400_700_10nm # 1000x31 * 31x3 = 1000x3
    #print('Ts:', ts)
    #ts = np.take(ts, np.argwhere(ts[:, 1] > 32).flatten(), axis=0)
    tv = ts @ np.ones(3)
    xy = (ts.transpose() / tv).transpose()[:, 0:2]
    print(xy.shape)
    Y = 32
    cs = []
    for i in range(xy.shape[0]):
        x = xy[i, 0]
        y = xy[i, 1]
        #plt.plot(sps[i])
        xyz = colors.xyY_to_XYZ(x, y, Y) #colors.color(x*Y/y, Y, (1-x-y)*Y/y)
        srgb = colors.xyz_to_srgb(xyz)
        cs.append(srgb)
    plt.scatter(xy.transpose()[0], xy.transpose()[1], c=cs, s=5.0)
    plt.plot([colors.spectral_color_xy(l)[0] for l in np.linspace(380, 700, 65)],
            [colors.spectral_color_xy(l)[1] for l in np.linspace(380, 700, 65)], 'c')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Not enough params', file=sys.stderr)
        sys.exit(1)

    datasheet = profile.FilmProfile(sys.argv[1])

    dyes = datasheet.dye()

    dyes = np.vstack([
        utils.to_400_700_10nm(dyes[0]),
        utils.to_400_700_10nm(dyes[1]),
        utils.to_400_700_10nm(dyes[2])
    ])
    
    light = illum.D55
    light = utils.to_400_700_10nm(light)
    dyes_gamut(dyes, light)

    #import wthanson

    #br = wthanson.Brewer(datasheet, light, extract_bases=False)
    #srgb = colors.color(0.21450662, 0.77336032, 0.16702399)
    #xyz = colors.srgb_to_xyz(srgb) 
    #qs = br.dyes_for_color(xyz)
    #print(qs)
    #dyes_gamut(dyes, light, qs.reshape((3, 1)))



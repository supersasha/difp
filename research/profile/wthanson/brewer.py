import numpy as np
import argparse
from scipy.interpolate import CubicSpline
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

from film_profile import FilmProfile
import illum
import data
import utils
import colors
import spectrum
import time

def exposure(logsense, sp):
    return np.dot(10 ** logsense, sp)

def exposure_density(logsense, sp0):
    sp = np.clip(sp0, 1e-10, None)
    return -np.log10(exposure(logsense, sp))

def normalized_sense(logsense, light):
    '''
    Normalized sense for the illuminant
    '''
    E = exposure(logsense, light)
    theta = -np.log10(E)
    ns =  (logsense.transpose() + theta).transpose()
    return ns

def transmittance(dyes, q):
    #return 1.0 / 10.0**(dyes[0]*q[0] + dyes[1]*q[1] + dyes[2]*q[2])
    return 10.0 ** -(dyes.transpose() @ q)

def outflux(dyes, light, q):
    return light * transmittance(dyes, q)

def dye_density(dyes, light, qs):
    '''
    Dye density

    Parameters
    ----------

    dyes
        The dyes
    light
        The illuminant used
    qs
        Quantities of the components

    Returns
    -------

    The density the components create
    in the quantities specified
    under the illuminant specified

    '''
    out = outflux(dyes, light, qs)
    return np.log10(np.sum(light) / np.sum(out))

def normalized_dyes(dyes, light, density=1.0, quantities=False):
    """
        Returns dyes that form together density of 'density'
          and at the same time are neutral for the light 'light'
    """
    trans_to_xyz_mtx = spectrum.transmittance_to_xyz_mtx(light)
    wp = illum.white_point(light)
    def f(qs):
        d = dye_density(dyes, light, qs)
        trans = transmittance(dyes, qs)
        xyz = trans @ trans_to_xyz_mtx.transpose()
        xy = colors.chromaticity(xyz)
        return (d - density)**2 + np.sum((xy - wp)**2)
    bounds = [(0.0, 3.0)] * 3
    r = opt.minimize(f, np.zeros(3), bounds=bounds)
    if quantities:
        return r.x
    return np.diag(r.x).dot(dyes)

def complementary_indices(idx):
    if idx == 0:
        return 1, 2
    elif idx == 1:
        return 0, 2
    else:
        return 0, 1

# Brewer seems need to be reworked: E.N.D. should be calculated in terms
# of dye quantities. This means that we need normalize dyes first so that
# they form together neutral density of 1. Then E.N.D. seems to be the
# simple dyes quantities of such normalized dyes.
class Brewer:
    def __init__(self, logsense, dyes, dev_light, proj_light):
        # Let dev_light be D55 for now, since we use D55 spectral bases
        self.dev_light = dev_light 
        self.proj_light = proj_light

        self.dyes = dyes #normalized_dyes(dyes, self.proj_light)
        self.logsense = logsense #normalized_sense(sense, self.dev_light)
        
        self.trans_to_xyz_mtx = spectrum.transmittance_to_xyz_mtx(self.proj_light)

        #xs = np.linspace(0.0, 6.0, 20)
        #self.end_to_q = [ CubicSpline(
        #    np.array([self.e_n_d_comp(idx, q) for q in xs]), xs
        #) for idx in range(3) ]

        self.refl_gen = spectrum.Spectrum(self.dev_light)

        #self.refl_gen.extract_bases()
        #self.refl_gen.savemat('d55-bases.mat')
        self.refl_gen.loadmat('d55-bases.mat')

    def dyes_for_color(self, xyz):
        '''
            Dyes for color

            Parameters
            ----------

            xyz
                color to match in XYZ coordinates

            Returns
            -------
            
            3-vector of dyes quantities that match the color
        '''

        def f(q):
            trans = transmittance(self.dyes, q)
            xyz1 = spectrum.transmittance_to_xyz(self.trans_to_xyz_mtx, trans)
            return colors.delta_E76_xyz(xyz, xyz1)

        bounds = [(0.0, 3.0)] * 3
        t0 = time.time()
        r = opt.dual_annealing(f, bounds, maxiter=200)
        #r = opt.minimize(f, np.array([0.0, 0.0, 0.0]), bounds=bounds)
        #print('Time:', time.time() - t0)
       
        xyz1 = self.qs_to_xyz(r.x)
        #print('xyz_out:', xyz1)
        #print(
        #    'Color dist:', r.fun,
        #    'srgb_in:', colors.xyz_to_srgb(xyz),
        #    'srgb_out:', colors.xyz_to_srgb(xyz1),
        #    'xy_in:', colors.chromaticity(xyz),
        #    'xy_out:', colors.chromaticity(xyz1),
        #)
        return r.x

    #def e_n_d(self, xyz):
    #    qs = self.dyes_for_color(xyz)
    #    return np.array([
    #        self.e_n_d_comp(0, qs[0]),
    #        self.e_n_d_comp(1, qs[1]),
    #        self.e_n_d_comp(2, qs[2])
    #    ])

    #def e_n_d_comp(self, comp_idx, q):
    #    '''
    #    Equivalent neutral density of a dye component
    #    
    #    Parameters
    #    ----------

    #    comp_idx: int
    #        Dye component index
    #    q
    #        The quantity of the component

    #    Returns
    #    -------

    #    The density of the overall dye when the component
    #    is coupled with other two components in quantities
    #    to create neutral color in the sense of
    #    the illuminant used
    #    '''
    #    ci1, ci2 = complementary_indices(comp_idx)
    #    wp = illum.white_point(self.proj_light)

    #    def f(x):
    #        qs = np.array([0.0, 0.0, 0.0])
    #        qs[comp_idx] = q
    #        qs[ci1] = x[0]
    #        qs[ci2] = x[1] 
    #        trans = transmittance(self.dyes, qs)
    #        # convert to xyz, then to chromaticity
    #        xyz = spectrum.transmittance_to_xyz(self.trans_to_xyz_mtx, trans)
    #        xy = colors.chromaticity(xyz)
    #        # compare chromaticity to the whitepoint of the illuminant
    #        d = xy - wp
    #        # return the squares of difference
    #        return np.sum(d*d)

    #    bounds = [(0, 10)] * 2
    #    r = opt.dual_annealing(f, bounds, maxiter=100)
    #    #r = opt.minimize(f, np.zeros(2), bounds=bounds, method='trust-constr')

    #    #print('MSE:', r.fun)
    #    qs = np.array([0.0, 0.0, 0.0])
    #    qs[comp_idx] = q
    #    qs[ci1] = r.x[0]
    #    qs[ci2] = r.x[1]
    #    return dye_density(self.dyes, self.proj_light, qs)

    #def end_to_qs(self, end):
    #    return np.array([ self.end_to_q[idx](end[idx]) for idx in range(3) ])

    def qs_to_xyz(self, qs):
        trans = transmittance(self.dyes, qs)
        return spectrum.transmittance_to_xyz(self.trans_to_xyz_mtx, trans)

    #def end_to_xyz(self, end):
    #    return self.qs_to_xyz(self.end_to_qs(end))

    def gen_colors(self):
        cols = []
        ends = []
        expds = []
        xyzs = [
            colors.color(12.08, 19.77, 16.28),
            colors.color(20.86, 12.00, 17.97),
            colors.color(14.27, 19.77, 26.42),
            colors.color( 7.53,  6.55, 34.26),
            colors.color(64.34, 59.10, 59.87),
            colors.color(58.51, 59.10, 29.81),
            colors.color(37.93, 30.05,  4.98),
            colors.color(95.05, 100.0, 108.9) / 2,
            colors.color(95.05, 100.0, 108.9) / 4,
            colors.color(95.05, 100.0, 108.9) / 8,

            #colors.color(11.0, 20.0, 6.0),
            #colors.color(13.0, 9.0, 53.0),
        ]
        for xyz in xyzs:
            sp, refl = self.refl_gen.spectrum_of(xyz)
            expd = exposure_density(self.logsense, sp)
            end = self.dyes_for_color(xyz) #e_n_d(xyz)
            #print('E.N.D.:', end)

            cols.append(xyz)
            expds.append(expd)
            ends.append(end)
        self.cols = np.vstack(cols)
        self.ends = np.vstack(ends)
        self.expds = np.vstack(expds)

    def solve_with_masks(self):
        def f(x0):
            x = x0[0:9].reshape((3, 3))
            ends1 = x.dot(self.expds.transpose()).transpose() + x0[9:12]
            d = 0
            for row in range(self.ends.shape[0]):
                dd = colors.delta_E76_xyz(
                    #self.end_to_xyz(self.ends[row]),
                    self.cols[row],
                    self.qs_to_xyz(ends1[row]) #end_to_xyz(ends1[row])
                )
                d += dd*dd
            return d
        #bounds = [(-10.0, 10.0)] * 12
        p = (0.0, 3.0)
        n = (-1.0, 0.0)
        u = (-1.0, 1.0)
        bounds = [p, n, n,
                  n, p, n,
                  n, n, p,
                  u, u, u]
        def cb(x, f, ctx):
            print(x, f, ctx)
        #r = opt.dual_annealing(f, bounds, maxiter=1000, callback=cb)
        r = opt.minimize(f, np.zeros(12), bounds=bounds)
        #print(r.fun)
        self.Gammas = r.x[0:9].reshape((3, 3))
        self.Ks = r.x[9:12]
        #x = r.x[0:9].reshape((3, 3))

    def solve_no_masks(self):
        def f(x0):
            gs = np.diag(x0[0:3])
            ks = x0[3:6]
            ends1 = gs.dot(self.expds.transpose()).transpose() + ks
            d = 0
            for row in range(self.ends.shape[0]):
                dd = colors.delta_E76_xyz(
                    self.cols[row],
                    self.qs_to_xyz(ends1[row]) #end_to_xyz(ends1[row])
                )
                d += dd*dd
            return d
        p = (0.0, 3.0)
        n = (-1.0, 0.0)
        u = (-1.0, 1.0)
        bounds = [p, p, p, u, u, u]
        def cb(x, f, ctx):
            print(x, f, ctx)
        r = opt.minimize(f, np.zeros(6), bounds=bounds)
        print(r.fun)
        self.Gammas = np.diag(r.x[0:3])
        self.Ks = r.x[3:6]

    def demo(self):
        ends1 = self.Gammas.dot(self.expds.transpose()).transpose() + self.Ks
                                                                    # + r.x[9:12]
        #ends1 = np.clip(ends1, 0.0, 5.0)
        d = 0
        def rect(x, y, w, h, xyz):
            plt.gca().add_patch(
                Rectangle((x, y), w, h, color=colors.xyz_to_srgb(xyz)))

        for row in range(self.cols.shape[0]):
            xyz = self.cols[row]
            xyz1 = self.qs_to_xyz(ends1[row]) #end_to_xyz(ends1[row])
            di = colors.delta_E76_xyz(
                #self.end_to_xyz(self.ends[row]),
                xyz,
                xyz1 
            )
            rect(2*(row // 12), row % 12, 1, 1, xyz)
            rect(2*(row // 12)+1, row % 12, 1, 1, xyz1)
            d += di
            print('dE:', di, xyz, xyz1)
        for i in range(1, 10):
            for j in range(0, 10):
                rgb = colors.color(random.random(), random.random(), random.random())
                #rgb = colors.color(0, 0, (i*10+j)/100)
                xyz = colors.srgb_to_xyz(rgb)
                sp, refl = self.refl_gen.spectrum_of(xyz)
                expd = exposure_density(self.logsense, sp)
                end = self.Gammas.dot(expd) + self.Ks #r.x[9:12]
                xyz1 = self.qs_to_xyz(end) #end_to_xyz(end)
                rect(2*i,   j, 1, 1, xyz)
                rect(2*i+1, j, 1, 1, xyz1)
        print('Brewer avg dE error: ', d / self.ends.shape[0])
        print('Brewer gammas:\n', self.Gammas) #r.x[0:9].reshape((3, 3)))
        print('Brewer consts:', self.Ks) #r.x[9:12])
        plt.xlim(0, 20)
        plt.ylim(0, 12)
        plt.show()
        return d / self.ends.shape[0]

def brewer(sense, dyes, dev_light=None, proj_light=None):
    if dev_light is None or proj_light is None:
        d55 = utils.to_400_700_10nm(illum.D55)
        if dev_light is None:
            dev_light = d55
        if proj_light is None:
            proj_light = d55
    brwr = Brewer(sense, dyes, dev_light, proj_light)
    brwr.gen_colors()
    brwr.solve_with_masks()
    return brwr

# ------------------------------------------------

def arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('datasheet')
    
    return parser.parse_args()

if __name__ == '__main__':
    opts = arguments()
    light = utils.to_400_700_10nm(illum.D55)
    profile = FilmProfile(opts.datasheet, mode31=True)
    logsense = normalized_sense(profile.sense())
    dyes = normalized_dyes(profile.dye())
    brwr = brewer(logsense, dyes, dev_light=light, proj_light=light)
    brwr.demo()
    #dyes = normalized_dyes(profile.dye(), light)
    #plt.plot(dyes.transpose())
    #plt.show()

import math
import json
import time

import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator

from film_profile import FilmProfile
import brewer
import utils
import illum
import spectrum
import colors

def zigzag(x, gamma, ymax):
    if x >= 0:
        return ymax
    x0 = -ymax/gamma
    if x <= x0:
        return 0
    return gamma * (x - x0)

def zigzag_p(x, gamma, ymax):
    if x < 0:
        return 0
    y = x * gamma
    #if y > ymax:
    #    return ymax
    return y

class Hanson:
    def __init__(self, film, paper, max_density=1.0, paper_gammas=1.0):#/1.2793):
        self.film = film
        self.paper = paper
        self.paper_gammas = np.array([paper_gammas] * 3)
        #self.paper_gammas = np.array([1/1.2793, 1.9809, 1.0])
        self.dev_light = utils.to_400_700_10nm(illum.D55)
        self.proj_light = utils.to_400_700_10nm(illum.D55)
        self.refl_light = utils.to_400_700_10nm(illum.D65)
        self.max_density = max_density #math.log10(256.0)

        self.film_sense = brewer.normalized_sense(film.sense(), self.dev_light)
        self.film_dyes = brewer.normalized_dyes(film.dye(), self.proj_light)
        self.film_max_qs = brewer.normalized_dyes(self.film_dyes, self.proj_light,
                            density=self.max_density, quantities=True)
        #print('Max qs:', self.film_max_qs)
        self.paper_dyes = brewer.normalized_dyes(paper.dye(), self.refl_light)
        # paper_sense calculated later

        self.mtx_refl = spectrum.transmittance_to_xyz_mtx(self.refl_light)
        self.mtx_proj = spectrum.transmittance_to_xyz_mtx(self.proj_light)
        
        self.refl_gen = spectrum.Spectrum(self.dev_light)
        self.refl_gen.loadmat('d55-bases.mat')
        #with open('d55-bases.json', 'w') as f:
        #    print(self.refl_gen.tojson(), file=f)
        self.brwr = brewer.brewer(self.film_sense, self.paper_dyes,
                                dev_light=self.dev_light, proj_light=self.proj_light)

    def compute_couplers(self):
        t0 = time.time()
        self.neg_gammas = self.brwr.Gammas.transpose().dot(np.diag(1/self.paper_gammas))
        dyes = np.diag(self.film_max_qs) @ self.film_dyes
        self.ref_waves = np.argmax(dyes, axis=1)
        intrp = PchipInterpolator
        xs = np.arange(31)
        couplers = []
        for idx in range(3):
            dye = dyes[idx]
            idx1, idx2 = brewer.complementary_indices(idx)
            coupler_at = [0, 0, 0]
            coupler_at[idx1] = dye[self.ref_waves[idx1]] - self.neg_gammas[idx, idx1] / self.neg_gammas[idx, idx] * dye[self.ref_waves[idx]]
            coupler_at[idx2] = dye[self.ref_waves[idx2]] - self.neg_gammas[idx, idx2] / self.neg_gammas[idx, idx] * dye[self.ref_waves[idx]]
            
            spl = intrp(
                [-5, self.ref_waves[2], self.ref_waves[1], self.ref_waves[0], 36],
                [0, coupler_at[2], coupler_at[1], coupler_at[0], 0]
            )
            couplers.append(spl(xs))
        self.couplers = np.vstack(couplers)
        self.neg_white = brewer.transmittance(dyes, np.ones(3)) * self.proj_light
        self.paper_sense = brewer.normalized_sense(self.paper.sense(), self.neg_white)
        print('Time:', time.time() - t0)

    def develop_film(self, H):
        dyes = np.diag(self.film_max_qs) @ self.film_dyes
        r = dyes[0, self.ref_waves[0]]
        g = dyes[1, self.ref_waves[1]]
        b = dyes[2, self.ref_waves[2]]
        #print('r,g,b:', [r, g, b])
        dev = np.array([
                zigzag(H[0], self.neg_gammas[0, 0], r),#self.film_max_qs[0]),
                zigzag(H[1], self.neg_gammas[1, 1], g),#self.film_max_qs[1]),
                zigzag(H[2], self.neg_gammas[2, 2], b),#self.film_max_qs[2]),
            ])
        dyes1 = np.diag([1/r, 1/g, 1/b]) @ dyes
        developed_dyes = np.diag(dev) @ dyes1
        developed_couplers = np.diag([
                1 - dev[0] / r, #self.film_max_qs[0],
                1 - dev[1] / g, #self.film_max_qs[1],
                1 - dev[2] / b, #self.film_max_qs[2],
            ]) @ self.couplers
        developed = developed_dyes + developed_couplers
        return developed

    def develop_paper(self, H):
        negative = self.develop_film(H)
        #plt.plot(negative.transpose())
        trans = brewer.transmittance(negative, np.ones(3)) * self.proj_light
        H1 = np.log10((10.0**self.paper_sense) @ trans) * self.paper_gammas #+ self.brwr.Ks
        print('H1:', H1)
        developed = np.diag(H1) @ self.paper_dyes
        return developed

    def H_neg(self):
        hsr = []
        hsg = []
        hsb = []
        xs = np.linspace(-2.0, 1.0, 100)
        for h in xs:
            negative = self.develop_film(np.array([0., h, 0.]))
            trans = brewer.transmittance(negative, np.ones(3)) * self.proj_light
            h1 = np.log10((10.0**self.paper_sense) @ trans)
            hsr.append(h1[0])
            hsg.append(h1[1])
            hsb.append(h1[2])
        plt.plot(xs, hsr, 'r')
        plt.plot(xs, hsg, 'g')
        plt.plot(xs, hsb, 'b')
        plt.show()

    def H_neg_of_H(self, H):
        t0 = time.time()
        negative = self.develop_film(H)
        trans = brewer.transmittance(negative, np.ones(3)) * self.proj_light
        res = np.log10((10.0**self.paper_sense) @ trans)
        print('Time H_neg_of_H:', time.time() - t0)
        return res

    def test_one(self, idx):
        d1 = np.array([0., 0., 0.])
        d1[idx] = -1.0
        developed1 = self.develop_paper(d1)
        developed0 = self.develop_paper(np.array([0., 0., 0.]))
        colors = ['r', 'g', 'b']
        g = developed0[idx, self.ref_waves] - developed1[idx, self.ref_waves]
        print(f'G:[{idx}]', -g)
        plt.plot(developed1[idx], colors[idx] + '--')
        plt.plot(developed0[idx], colors[idx])

    def test(self):
        for i in range(3):
            self.test_one(i)
        print('Neg gammas:')
        print(self.neg_gammas)
        print('Gammas:')
        print(self.brwr.Gammas)
        plt.show()

    def develop(self, xyz, debug=False):
        sp, refl = self.refl_gen.spectrum_of(xyz)
        H = np.log10(brewer.exposure(self.film_sense, sp))

        #if debug:
        #    zs = []
        #    xs = np.linspace(-20.0, 5.0, 100)
        #    for x in xs:
        #        zs.append(zigzag(x, self.neg_gammas[0, 0], self.film_max_qs[0]))
        #    plt.plot(xs, zs)
        #    plt.show()

        #dev = np.array([
        #        zigzag(H[0], self.neg_gammas[0, 0], self.film_max_qs[0]),
        #        zigzag(H[1], self.neg_gammas[1, 1], self.film_max_qs[1]),
        #        zigzag(H[2], self.neg_gammas[2, 2], self.film_max_qs[2]),
        #    ])

        #if debug:
        #    print('H:', H, 'dev:', dev)

        #developed_dyes = np.diag(dev) @ self.film_dyes
        #developed_couplers = np.diag([
        #        1 - dev[0] / self.film_max_qs[0],
        #        1 - dev[1] / self.film_max_qs[1],
        #        1 - dev[2] / self.film_max_qs[2],
        #    ]) @ self.couplers
        #developed = developed_dyes + developed_couplers

        developed = self.develop_film(H)

        trans = brewer.transmittance(developed, np.ones(3)) * self.proj_light
        #if debug:
        #    plt.plot(trans.transpose())
        #    plt.show()
        H1 = np.log10((10.0**self.paper_sense) @ trans) * self.paper_gammas #+ self.brwr.Ks
        if debug:
            print('H1:', H1)
        developed = np.diag(H1) @ self.paper_dyes
        
        #plt.ylim(-0.1, 6)
        #plt.plot(developed_dyes[0], 'c--')
        #plt.plot(developed_couplers[0], 'r')
        #plt.plot(developed[0], 'c')
        #
        #plt.plot(developed_dyes[1], 'm--')
        #plt.plot(developed_couplers[1], 'g')
        #plt.plot(developed[1], 'm')
        #
        #plt.plot(developed_dyes[2], 'y--')
        #plt.plot(developed_couplers[2], 'b')
        #plt.plot(developed[2], 'y')
        #plt.show()

        trans = brewer.transmittance(developed, np.ones(3))
        xyz1 = spectrum.transmittance_to_xyz(self.mtx_refl, trans)
        return xyz1

    def develop_img_srgb(self, filename):
        img_in = plt.imread(filename) / 255
        img = img_in.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                c = img[i, j]
                c = colors.srgb_to_xyz(c)
                c = self.develop(c)
                img[i, j] = colors.xyz_to_srgb(c)
        img.clip(0, 1)
        plt.imsave('pics/result.jpg', img)
        plt.subplot(121)
        plt.imshow(img_in)
        plt.subplot(122)
        plt.imshow(img)
        plt.show()

    def to_json(self):
        obj = {
            'film_sense': self.film_sense.tolist(), # 3x31
            'film_dyes': self.film_dyes.tolist(), # 3x31
            'paper_sense': self.paper_sense.tolist(), # 3x31
            'paper_dyes': self.paper_dyes.tolist(), # 3x31
            'couplers': self.couplers.tolist(), # 3x31
            'proj_light': self.proj_light.tolist(), # 31
            'mtx_refl': self.mtx_refl.tolist(), # 3x31
            'neg_gammas': self.neg_gammas.diagonal().tolist(), # 3
            'paper_gammas': self.paper_gammas.tolist(), # 3
            'film_max_qs': self.film_max_qs.tolist() # 3
        }
        return json.dumps(obj, indent=4)

def arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    def add_common_opts(subparser):
        subparser.add_argument('-f', '--film', help='Film datasheet',
            default='../../../profiles/datasheets/kodak-vision-250d-5207.datasheet')
        subparser.add_argument('-p', '--paper', help='Paper datasheet',
            default='../../../profiles/datasheets/kodak-endura.datasheet')

    parser_mkprof = subparsers.add_parser('mkprof')
    add_common_opts(parser_mkprof)

    parser_debug = subparsers.add_parser('debug')
    add_common_opts(parser_debug)

    parser_demo = subparsers.add_parser('demo')
    add_common_opts(parser_demo)
    parser_demo.add_argument('image_path')
    
    return parser.parse_args()

def cmd_mkprof(opts):
    film = FilmProfile(opts.film, mode31=True)
    paper = FilmProfile(opts.paper, mode31=True)
    hanson = Hanson(film, paper)
    hanson.compute_couplers()
    print(hanson.to_json())
    #print(hanson.refl_gen.to_json())

def cmd_demo(opts):
    film = FilmProfile(opts.film, mode31=True)
    paper = FilmProfile(opts.paper, mode31=True)
    hanson = Hanson(film, paper, max_density=1.0, paper_gammas=2.0)
    hanson.compute_couplers()
    hanson.develop_img_srgb(opts.image_path)

def cmd_debug(opts):
    film = FilmProfile(opts.film, mode31=True)
    paper = FilmProfile(opts.paper, mode31=True)
    hanson = Hanson(film, paper, max_density=1.0, paper_gammas=2.0)
    hanson.compute_couplers()
    #hanson.test()
    H = np.array([-1.0, 0.0, 0.0])
    #print(f'H_neg({H}) = ', hanson.H_neg_of_H(H))
    hanson.H_neg()
    print('Gammas:')
    print(hanson.brwr.Gammas)
    print('Neg gammas:')
    print(hanson.neg_gammas)

    #for x in np.linspace(0, 1, 10):
    #    srgb = colors.color(x, x, x)
    #    xyz = colors.srgb_to_xyz(srgb)
    #    xyz1 = hanson.develop(xyz, debug=True)
    #    srgb1 = colors.xyz_to_srgb(xyz1)
    #    print(srgb1)
    #plt.show()

if __name__ == '__main__':
    opts = arguments()
    if opts.command == 'mkprof':
        cmd_mkprof(opts)
    elif opts.command == 'debug':
        cmd_debug(opts)
    elif opts.command == 'demo':
        cmd_demo(opts)

import math
import json
import time
import sys
import random

import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator
import scipy.optimize as opt

from film_profile import FilmProfile
import brewer
import utils
import illum
import spectrum
import colors

import nlopt

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

def bell(q, x):
    # q=[a, mu, sigma]
    return q[0]*np.exp(-((x-q[1])/q[2])**2)

def bell1(a, mu, sigma, x):
    # q=[a, mu, sigma]
    return a*np.exp(-((x-mu)/sigma)**2)

def reference_colors():
    def srgb_to_xyz(r, g, b):
        return colors.srgb_to_xyz(colors.color(r, g, b))
    vs = np.linspace(0.1, 1, 5);
    reds = [(srgb_to_xyz(v, 0, 0), v * 4) for v in vs]
    greens = [(srgb_to_xyz(0, v, 0), 1) for v in vs]
    blues = [(srgb_to_xyz(0, 0, v), 1) for v in vs]

    cyans = [(srgb_to_xyz(0, v, v), v) for v in vs]
    magentas = [(srgb_to_xyz(v, 0, v), v) for v in vs]
    yellows = [(srgb_to_xyz(v, v, 0), v) for v in vs]

    grays = [(srgb_to_xyz(v, v, v), 2) for v in vs]
    xyzs = reds + greens + blues + grays + cyans + magentas + yellows
    return xyzs


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
        
        if film is not None:
            self.film_sense = brewer.normalized_sense(film.sense(), self.dev_light)
            #print("film_sense (norm):", self.film_sense)
            self.film_dyes = brewer.normalized_dyes(film.dye(), self.proj_light)
            self.film_max_qs = brewer.normalized_dyes(self.film_dyes, self.proj_light,
                                density=self.max_density, quantities=True)
            #print('Max qs:', self.film_max_qs)
        if paper is not None:
            self.paper_dyes = brewer.normalized_dyes(paper.dye(), self.refl_light)
            # paper_sense calculated later

        self.mtx_refl = spectrum.transmittance_to_xyz_mtx(self.refl_light)
        self.mtx_proj = spectrum.transmittance_to_xyz_mtx(self.proj_light)
        
        #self.refl_gen = spectrum.Spectrum(self.dev_light)
        #self.refl_gen.loadmat('d55-bases.mat')
        #with open('spectra2/spectrum-d55-4.json') as f:
        #    self.refl_gen = spectrum.Spectrum(f.read())
        self.refl_gen = spectrum.load_spectrum('spectra2/spectrum-d55-4.json')

        #self.brwr = brewer.brewer(self.film_sense, self.paper_dyes,
        #                        dev_light=self.dev_light, proj_light=self.proj_light)

    #def compute_couplers(self):
    #    t0 = time.time()
    #    self.neg_gammas = self.brwr.Gammas.transpose().dot(np.diag(1/self.paper_gammas))
    #    dyes = np.diag(self.film_max_qs) @ self.film_dyes
    #    self.ref_waves = np.argmax(dyes, axis=1)
    #    intrp = PchipInterpolator
    #    xs = np.arange(31)
    #    #couplers_at = np.array([
    #    #    [0.0, 0.45, 0.08],
    #    #    [0.75, 0.0, 0.26],
    #    #    [0.08, 0.6, 0.0],
    #    #])
    #    couplers_at = np.array([
    #        [0.0, 0.45, 0.08],
    #        [0.75, 0.0, 0.26],
    #        [0.08, 0.6, 0.0],
    #    ])
    #    couplers = []
    #    for idx in range(3):
    #        dye = dyes[idx]
    #        idx1, idx2 = brewer.complementary_indices(idx)
    #        coupler_at = couplers_at[idx] #[0, 0, 0]
    #        #coupler_at[idx1] = dye[self.ref_waves[idx1]] - self.neg_gammas[idx, idx1] / self.neg_gammas[idx, idx] * dye[self.ref_waves[idx]]
    #        #coupler_at[idx2] = dye[self.ref_waves[idx2]] - self.neg_gammas[idx, idx2] / self.neg_gammas[idx, idx] * dye[self.ref_waves[idx]]
    #        
    #        spl = intrp(
    #            [-5, self.ref_waves[2], self.ref_waves[1], self.ref_waves[0], 36],
    #            [0, coupler_at[2], coupler_at[1], coupler_at[0], 0]
    #        )
    #        couplers.append(spl(xs))
    #    self.couplers = np.vstack(couplers)
    #    self.neg_white = brewer.transmittance(dyes, np.ones(3)) * self.proj_light
    #    self.paper_sense = brewer.normalized_sense(self.paper.sense(), self.neg_white)
    #    print('Time:', time.time() - t0)
    
    def make_couplers(self, q):
        dyes = np.diag(self.film_max_qs) @ self.film_dyes
        #self.ref_waves = np.argmax(dyes, axis=1)
        #self.ref_waves = self.ref_waves * 10 + 400
        #intrp = PchipInterpolator
        couplers = []
        xs = np.arange(31)*10.0 + 400
        for idx in range(3): #(3):
            b = idx*15
            # + bell(q[3+b+3:3+b+6], xs)
            # + bell(q[3+b+3:3+b+6], xs) + bell(q[3+b+6:3+b+9], xs)
            couplers.append(bell(q[3+b:3+b+3], xs) + bell(q[3+b+3:3+b+6], xs) + bell(q[3+b+6:3+b+9], xs) + bell(q[3+b+9:3+b+12], xs) + bell(q[3+b+12:3+b+15], xs))
        #couplers.append(bell1(q[3], self.ref_waves[1], q[4], xs) + bell1(q[5], self.ref_waves[2], q[6], xs))
        #couplers.append(bell1(q[7], self.ref_waves[0], q[8], xs) + bell1(q[9], self.ref_waves[2], q[10], xs))
        #couplers.append(bell1(q[11], self.ref_waves[0], q[12], xs) + bell1(q[13], self.ref_waves[1], q[14], xs))

        #couplers.append(np.zeros(31))
        self.couplers = np.vstack(couplers)
        self.neg_white = brewer.transmittance(dyes, np.ones(3)) * self.proj_light
        self.paper_sense = brewer.normalized_sense(self.paper.sense(), self.neg_white)

    def solve(self):
        #xyzs = [
        #    colors.color(12.08, 19.77, 16.28),
        #    colors.color(20.86, 12.00, 17.97),
        #    colors.color(14.27, 19.77, 26.42),
        #    colors.color( 7.53,  6.55, 34.26),
        #    colors.color(64.34, 59.10, 59.87),
        #    colors.color(58.51, 59.10, 29.81),
        #    colors.color(37.93, 30.05,  4.98),
        #    colors.color(95.05, 100.0, 108.9),
        #    colors.color(95.05, 100.0, 108.9) / 4,
        #    colors.color(95.05, 100.0, 108.9) / 16,
        #    colors.color(95.05, 100.0, 108.9) / 64,
        #    colors.color(95.05, 100.0, 108.9) / 256,
        #        
        #    colors.color(45.0, 27.0, 10.0),
        #    colors.color(41.0, 21.0, 2.0) / 1,
        #    colors.color(40.0, 21.5, 2.1) / 2,
        #    colors.color(41.5, 22.0, 1.8) / 4,

        #    #colors.color(11.0, 20.0, 6.0),
        #    #colors.color(13.0, 9.0, 53.0),
        #]

        #xyzs = []

        #def rndc(a, b):
        #    return a + random.random() * (b-a)
        
        #xyzs = [
        #    colors.srgb_to_xyz(colors.color(1, 0, 0)),
        #    colors.srgb_to_xyz(colors.color(0.9, 0, 0)),
        #    colors.srgb_to_xyz(colors.color(0.8, 0, 0)),
        #    colors.srgb_to_xyz(colors.color(0.7, 0, 0)),
        #    colors.srgb_to_xyz(colors.color(0.6, 0, 0)),
        #    colors.srgb_to_xyz(colors.color(0.5, 0, 0)),
        #    colors.srgb_to_xyz(colors.color(0, 1, 0)),
        #    colors.srgb_to_xyz(colors.color(0, 0.9, 0)),
        #    colors.srgb_to_xyz(colors.color(0, 0.8, 0)),
        #    colors.srgb_to_xyz(colors.color(0, 0.7, 0)),
        #    colors.srgb_to_xyz(colors.color(0, 0.6, 0)),
        #    colors.srgb_to_xyz(colors.color(0, 0.5, 0)),
        #    colors.srgb_to_xyz(colors.color(0, 0, 1)),
        #    colors.srgb_to_xyz(colors.color(0, 0, 0.9)),
        #    colors.srgb_to_xyz(colors.color(0, 0, 0.8)),
        #    colors.srgb_to_xyz(colors.color(0, 0, 0.7)),
        #    colors.srgb_to_xyz(colors.color(0, 0, 0.6)),
        #    colors.srgb_to_xyz(colors.color(0, 0, 0.5)),
        #    colors.srgb_to_xyz(colors.color(0, 1, 1)),
        #    colors.srgb_to_xyz(colors.color(0, 0.9, 0.9)),
        #    colors.srgb_to_xyz(colors.color(0, 0.8, 0.8)),
        #    colors.srgb_to_xyz(colors.color(0, 0.7, 0.7)),
        #    colors.srgb_to_xyz(colors.color(0, 0.6, 0.6)),
        #    colors.srgb_to_xyz(colors.color(0, 0.5, 0.5)),
        #    colors.srgb_to_xyz(colors.color(1, 0, 1)),
        #    colors.srgb_to_xyz(colors.color(0.9, 0, 0.9)),
        #    colors.srgb_to_xyz(colors.color(0.8, 0, 0.8)),
        #    colors.srgb_to_xyz(colors.color(0.7, 0, 0.7)),
        #    colors.srgb_to_xyz(colors.color(0.6, 0, 0.6)),
        #    colors.srgb_to_xyz(colors.color(0.5, 0, 0.5)),
        #    colors.srgb_to_xyz(colors.color(1, 1, 0)),
        #    colors.srgb_to_xyz(colors.color(0.9, 0.9, 0)),
        #    colors.srgb_to_xyz(colors.color(0.8, 0.8, 0)),
        #    colors.srgb_to_xyz(colors.color(0.7, 0.7, 0)),
        #    colors.srgb_to_xyz(colors.color(0.6, 0.6, 0)),
        #    colors.srgb_to_xyz(colors.color(0.5, 0.5, 0)),
        #    colors.srgb_to_xyz(colors.color(1, 1, 1)),
        #    colors.srgb_to_xyz(colors.color(0.8, 0.8, 0.8)),
        #    colors.srgb_to_xyz(colors.color(0.5, 0.5, 0.5)),
        #    colors.srgb_to_xyz(colors.color(0.25, 0.25, 0.25)),
        #    colors.srgb_to_xyz(colors.color(0.1, 0.1, 0.1)),
        #    colors.srgb_to_xyz(colors.color(0.05, 0.05, 0.05)),
        #]
        
        xyzs = reference_colors()

        #for i in range(100):
        #    c = colors.color(rndc(0, 1), rndc(0, 1), rndc(0, 1))
        #    xyzs.append(colors.srgb_to_xyz(c))
        #for r in np.linspace(0, 1, 5):
        #    for g in np.linspace(0, 1, 5):
        #        for b in np.linspace(0, 1, 5):
        #            c = colors.color(r, g, b)
        #            xyzs.append(colors.srgb_to_xyz(c))

        def f(x):
            self.make_couplers(x)
            #print("couplers: ", self.couplers)
            d = 0
            for xyz, v in xyzs:
                #print("xyz:", xyz)
                xyz1 = self.develop(xyz, x)
                #print("xyz1:", xyz1)
                d0 = colors.delta_E76_xyz(xyz, xyz1) * v
                d += d0*d0
            return d
        #x0 = np.array([0.5, 0.5, 0.5,
        #                1, 500, 100,
        #                1, 500, 100,
        #                1, 500, 100,
        #                1, 500, 100,
        #                1, 500, 100,
        #                1, 500, 100,
        #                1, 500, 100,
        #                1, 500, 100,
        #                1, 500, 100,
        #                1, 500, 100,
        #                1, 500, 100,
        #                1, 500, 100,
        #                1, 500, 100,
        #                1, 500, 100,
        #                1, 500, 100])
        #print("of:", f(x0))
        #return
        
        gmin = 1.0e10
        step = 0
        def f_nl(x, grad):
            nonlocal step
            nonlocal gmin
            step += 1
            g = f(x)
            if g < gmin:
                gmin = g
                print(step, g, file=sys.stderr)
            return g

        #bounds = [(0.0001, 2.0)] * 21
        bounds = [(0.0001, 1.0)]*3 + [(0.0001, 3), (350.0, 750.0), (20.0, 200.0)]*15 #(20.0, 200.0)]*3
        #bounds = [(0.0001, 1.0)]*3 + [(0.0001, 3), (20.0, 100.0)]*6

         
        def cb(x, f, ctx):
            #with np.set_printoptions(precision=3, suppress=True):
            print(f, x.reshape((10, 3)), ctx, file=sys.stderr)
        #r = opt.dual_annealing(f, bounds, maxiter=1000, callback=cb)
        #print(r, file=sys.stderr)
        #self.solution = r.x


        # GN_CRS2_LM
        optm = nlopt.opt(nlopt.GN_ISRES, len(bounds))
        optm.set_min_objective(f_nl)
        lb = np.array(list(list(zip(*bounds))[0]))
        ub = np.array(list(list(zip(*bounds))[1]))
        optm.set_lower_bounds(lb)
        optm.set_upper_bounds(ub)
        optm.set_maxtime(300) #(3600 * 3)
        #optm.set_population(80)
        #x0 = np.array([rndc(l, u) for (l, u) in bounds])
        xopt = optm.optimize(lb) #((lb + ub)/2)
        self.solution = xopt

         

        #r = opt.minimize(f, np.array(list(list(zip(*bounds))[0])) + np.array(list(list(zip(*bounds))[1])), bounds=bounds, method='L-BFGS-B',
        #        options={'maxiter': 2000, 'maxfun': 1000000})
        
        #print(r, file=sys.stderr)
        #self.solution = r.x
        
    def develop(self, xyz, q):
        sp, refl = self.refl_gen.spectrum_of(xyz)

        H = np.log10(brewer.exposure(self.film_sense, sp))
        #print("H:", H)

        negative = self.develop_film(H, q)
        positive = self.develop_paper(negative)

        trans = brewer.transmittance(positive, np.ones(3))
        xyz1 = spectrum.transmittance_to_xyz(self.mtx_refl, trans)
        return xyz1

    def develop_film(self, H, q):
        dyes = np.diag(self.film_max_qs) @ self.film_dyes
        dev = np.array([
                zigzag(H[0], q[0], 1.0),
                zigzag(H[1], q[1], 1.0),
                zigzag(H[2], q[2], 1.0),
            ])
        #print("dev:", dev)
        #print("dyes:", dyes)
        developed_dyes = np.diag(dev) @ dyes
        developed_couplers = np.diag(1-dev) @ self.couplers
        developed = developed_dyes + developed_couplers
        return developed

    def develop_paper(self, negative):
        trans = brewer.transmittance(negative, np.ones(3))
        sp = trans * self.proj_light
        H1 = np.log10((10.0**self.paper_sense) @ sp) * self.paper_gammas #+ self.brwr.Ks
        developed = np.diag(H1) @ self.paper_dyes
        return developed

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

    #def develop(self, xyz, debug=False):
    #    sp, refl = self.refl_gen.spectrum_of(xyz)
    #    H = np.log10(brewer.exposure(self.film_sense, sp))

    #    developed = self.develop_film(H)

    #    trans = brewer.transmittance(developed, np.ones(3)) * self.proj_light
    #    H1 = np.log10((10.0**self.paper_sense) @ trans) * self.paper_gammas #+ self.brwr.Ks
    #    if debug:
    #        print('H1:', H1)
    #    developed = np.diag(H1) @ self.paper_dyes

    #    trans = brewer.transmittance(developed, np.ones(3))
    #    xyz1 = spectrum.transmittance_to_xyz(self.mtx_refl, trans)
    #    return xyz1

    def develop_img_srgb(self, filename):
        #q = np.array([0.2824192 , 0.02646818, 0.42294964,
        #              0.29478017, 0.17355421, 0.28623832,
        #              0.66749234, 0.7950928 , 0.85811659])
        #q = np.array([0.16112379, 0.01374971, 0.4990468 , 0.27349516, 0.2116357 ,
        #   0.24817832, 0.70480255, 0.72040886, 0.80619954])
        q = np.array([0.40700456, 0.0647628 , 0.35828811, 0.21178703, 0.02776719,
                    0.36165672, 0.26048918, 0.41501238, 0.35467305])
        img_in = plt.imread(filename) / 255
        img = img_in.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                c = img[i, j]
                c = colors.srgb_to_xyz(c)
                c = self.develop(c, q)
                img[i, j] = colors.xyz_to_srgb(c)
        img.clip(0, 1)
        plt.imsave('pics/result.jpg', img)
        plt.subplot(121)
        plt.imshow(img_in)
        plt.subplot(122)
        plt.imshow(img)
        plt.show()

    def to_json(self):
        dyes = np.diag(self.film_max_qs) @ self.film_dyes
        obj = {
            'film_sense': self.film_sense.tolist(), # 3x31
            'film_dyes': dyes.tolist(), # 3x31
            'paper_sense': self.paper_sense.tolist(), # 3x31
            'paper_dyes': self.paper_dyes.tolist(), # 3x31
            'couplers': self.couplers.tolist(), # 3x31
            'proj_light': self.proj_light.tolist(), # 31
            'dev_light': self.dev_light.tolist(), # 31
            'mtx_refl': self.mtx_refl.tolist(), # 3x31
            'neg_gammas': self.solution[:3].tolist(), #self.neg_gammas.diagonal().tolist(), # 3
            'paper_gammas': self.paper_gammas.tolist(), # 3
            'film_max_qs': self.film_max_qs.tolist() # 3
        }
        return json.dumps(obj, indent=4)

    def from_json(self, js):
        obj = json.loads(js)
        self.film_sense = np.array(obj['film_sense'])
        self.film_max_qs = np.array(obj['film_max_qs'])
        self.film_dyes = np.diag(1 / self.film_max_qs) @ np.array(obj['film_dyes'])
        self.paper_sense = np.array(obj['paper_sense'])
        self.paper_dyes = np.array(obj['paper_dyes'])
        self.couplers = np.array(obj['couplers'])
        self.paper_gammas = np.array(obj['paper_gammas'])
        return obj['neg_gammas']

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

    parser_analyze = subparsers.add_parser('analyze')
    parser_analyze.add_argument('profile', help='Profile', default=None)

    parser_debug = subparsers.add_parser('debug')
    add_common_opts(parser_debug)

    parser_demo = subparsers.add_parser('demo')
    add_common_opts(parser_demo)
    parser_demo.add_argument('image_path')
    
    return parser.parse_args()

def cmd_mkprof(opts):
    film = FilmProfile(opts.film, mode31=True)
    paper = FilmProfile(opts.paper, mode31=True)
    hanson = Hanson(film, paper, max_density=2.0, paper_gammas=2.0)
    hanson.solve()
    hanson.make_couplers(hanson.solution)
    print(hanson.to_json())
    plt.plot(hanson.couplers[0], 'r')
    plt.plot(hanson.couplers[1], 'g')
    plt.plot(hanson.couplers[2], 'b')
    plt.show()

def cmd_analyze(opts):
    hanson = Hanson(None, None, max_density=2.0, paper_gammas=2.0)
    q = None
    if opts.profile is not None:
        with open(opts.profile) as f:
            js = f.read()
            q = hanson.from_json(js)
    #xyzs = reference_colors()
    #d2sum = 0
    #dsum = 0
    #for xyz, v in xyzs:
    #    xyz1 = hanson.develop(xyz, q)
    #    srgb = colors.xyz_to_srgb(xyz)
    #    srgb1 = colors.xyz_to_srgb(xyz1)
    #    d = colors.delta_E76_xyz(xyz, xyz1)
    #    d2sum += d*d
    #    dsum += d
    #    print(f'{srgb} --> {srgb1}: {d}')
    #print(d2sum, dsum, dsum/len(xyzs))
    xs = np.linspace(400, 700, 31)
    plt.plot(xs, hanson.couplers[0], 'r')
    plt.plot(xs, hanson.couplers[1], 'g')
    plt.plot(xs, hanson.couplers[2], 'b')
    plt.plot(xs, hanson.film_dyes[0]*hanson.film_max_qs[0], 'c')
    plt.plot(xs, hanson.film_dyes[1]*hanson.film_max_qs[0], 'm')
    plt.plot(xs, hanson.film_dyes[2]*hanson.film_max_qs[0], 'y')
    plt.show()

def cmd_demo(opts):
    film = FilmProfile(opts.film, mode31=True)
    paper = FilmProfile(opts.paper, mode31=True)
    hanson = Hanson(film, paper, max_density=1.0, paper_gammas=5.0)
    #hanson.compute_couplers()
    #q = np.array([0.2824192 , 0.02646818, 0.42294964,
    #              0.29478017, 0.17355421, 0.28623832,
    #              0.66749234, 0.7950928 , 0.85811659])
    #q = np.array([0.16112379, 0.01374971, 0.4990468 , 0.27349516, 0.2116357 ,
    #   0.24817832, 0.70480255, 0.72040886, 0.80619954])
    q = np.array([0.40700456, 0.0647628 , 0.35828811, 0.21178703, 0.02776719,
                0.36165672, 0.26048918, 0.41501238, 0.35467305])
    hanson.make_couplers(q)
    hanson.develop_img_srgb(opts.image_path)

def cmd_debug(opts):
    film = FilmProfile(opts.film, mode31=True)
    paper = FilmProfile(opts.paper, mode31=True)
    hanson = Hanson(film, paper, max_density=1.0, paper_gammas=4.0)
    hanson.solve()

    #hanson.compute_couplers()
    #print('Ref waves:', hanson.ref_waves, hanson.ref_waves * 5 + 400)
    ##hanson.test()
    ##H = np.array([-1.0, 0.0, 0.0])
    ##print(f'H_neg({H}) = ', hanson.H_neg_of_H(H))
    ##H = np.array([0.0, -1.0, 0.0])
    ##print(f'H_neg({H}) = ', hanson.H_neg_of_H(H))
    ##H = np.array([0.0, 0.0, -1.0])
    ##print(f'H_neg({H}) = ', hanson.H_neg_of_H(H))
    #print('Gammas:')
    #print(hanson.brwr.Gammas)
    #print('Neg gammas:')
    #print(hanson.neg_gammas)
    ##hanson.H_neg()
    #hs = np.linspace(-2, 1, 100)
    #h1s = []
    #for h in hs:
    #    h1s.append(hanson.H_neg_of_H(np.array([h, 0, 0])))
    #h1s = np.array(h1s).transpose()
    #plt.plot(hs, h1s[0], 'r')
    #plt.plot(hs, h1s[1], 'g')
    #plt.plot(hs, h1s[2], 'b')
    #plt.show()

        

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
    elif opts.command == 'analyze':
        cmd_analyze(opts)
    elif opts.command == 'debug':
        cmd_debug(opts)
    elif opts.command == 'demo':
        cmd_demo(opts)

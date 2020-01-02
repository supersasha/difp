import math
import json

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
    def __init__(self, film, paper, paper_gammas=[5.0, 5.0, 5.0]):
        self.film = film
        self.paper = paper
        self.paper_gammas = np.array(paper_gammas)
        self.dev_light = utils.to_400_700_10nm(illum.D55)
        self.proj_light = utils.to_400_700_10nm(illum.D55)
        self.refl_light = utils.to_400_700_10nm(illum.D65)
        self.max_density = 1.0 #math.log10(256.0)

        self.film_sense = brewer.normalized_sense(film.sense(), self.dev_light)
        self.film_dyes = brewer.normalized_dyes(film.dye(), self.proj_light)
        self.film_max_qs = brewer.normalized_dyes(self.film_dyes, self.proj_light,
                            density=self.max_density, quantities=True)
        self.paper_dyes = brewer.normalized_dyes(paper.dye(), self.refl_light)
        # paper_sense calculated later

        self.mtx_refl = spectrum.transmittance_to_xyz_mtx(self.refl_light)
        self.mtx_proj = spectrum.transmittance_to_xyz_mtx(self.proj_light)
        
        self.refl_gen = spectrum.Spectrum(self.dev_light)
        self.refl_gen.loadmat('d55-bases.mat')
        #with open('d55-bases.json', 'w') as f:
        #    print(self.refl_gen.tojson(), file=f)


    def compute_couplers(self):
        self.brwr = brewer.brewer(self.film_sense, self.paper_dyes,
                                dev_light=self.dev_light, proj_light=self.proj_light)
        self.neg_gammas = self.brwr.Gammas.transpose().dot(np.diag(1/self.paper_gammas))
        #print('Gammas:\n', self.brwr.Gammas)
        #print('Negative gammas:\n', self.neg_gammas)
        dyes = np.diag(self.film_max_qs) @ self.film_dyes

        #trans = brewer.transmittance(dyes, np.array([0, 0, 1]))
        #xyz1 = spectrum.transmittance_to_xyz(self.mtx_proj, trans)
        #print(f'Dyes xyz: {xyz1}, srgb: {colors.xyz_to_srgb(xyz1)} [{self.film_max_qs}]')
        #plt.plot(dyes.transpose())
        #plt.show()

        ref_waves = np.argmax(dyes, axis=1)
        #print('Ref waves: ', ref_waves, ref_waves*5+400)

        intrp = PchipInterpolator
        xs = np.arange(31)
        
        couplers = []
        for idx in range(3):
            k = dyes[idx, ref_waves[idx]] / self.neg_gammas[idx, idx]
            dye = dyes[idx] / k
     
            idx1, idx2 = brewer.complementary_indices(idx)
            
            coupler_at = [0, 0, 0]
            coupler_at[idx1] = k * (dye[ref_waves[idx1]] - self.neg_gammas[idx, idx1])
            coupler_at[idx2] = k * (dye[ref_waves[idx2]] - self.neg_gammas[idx, idx2])
            
            spl = intrp(
                [-5, ref_waves[2], ref_waves[1], ref_waves[0], 36],
                [0, coupler_at[2], coupler_at[1], coupler_at[0], 0]
            )
            couplers.append(spl(xs))
        self.couplers = np.vstack(couplers)
        
        #plt.plot(self.couplers[0], 'r')
        #plt.plot(self.couplers[1], 'g')
        #plt.plot(self.couplers[2], 'b')
        #plt.plot(dyes[0], 'c')
        #plt.plot(dyes[1], 'm')
        #plt.plot(dyes[2], 'y')
        #plt.show()
        
        self.neg_white = brewer.transmittance(dyes, np.ones(3)) * self.proj_light
        self.paper_sense = brewer.normalized_sense(paper.sense(), self.neg_white)
        self.paper_sense[0] -= 0.009
        self.paper_sense[1] += 0.003
        self.paper_sense[2] += 0.003
        #return self.couplers

    def develop(self, xyz):
        sp, refl = self.refl_gen.spectrum_of(xyz)
        H = np.log10(brewer.exposure(self.film_sense, sp))

        #zs = []
        #xs = np.linspace(-20.0, 5.0, 100)
        #for x in xs:
        #    zs.append(zigzag(x, self.neg_gammas[0, 0], self.film_max_qs[0]))
        #plt.plot(xs, zs)
        #plt.show()


        dev = np.array([
                zigzag(H[0], self.neg_gammas[0, 0], self.film_max_qs[0]),
                zigzag(H[1], self.neg_gammas[1, 1], self.film_max_qs[1]),
                zigzag(H[2], self.neg_gammas[2, 2], self.film_max_qs[2]),
            ])
        #print('H:', H, 'dev:', dev)
        developed_dyes = np.diag(dev) @ self.film_dyes
        developed_couplers = np.diag([
                1 - dev[0] / self.film_max_qs[0],
                1 - dev[1] / self.film_max_qs[1],
                1 - dev[2] / self.film_max_qs[2],
            ]) @ self.couplers
        developed = developed_dyes + developed_couplers

        trans = brewer.transmittance(developed, np.ones(3)) * self.proj_light
        #plt.plot(trans.transpose())
        H = np.log10((10.0**self.paper_sense) @ trans) * self.paper_gammas
        #print('H1:', H)
        developed = np.diag(H) @ self.paper_dyes
        #developed = np.diag([
        #        zigzag(D[0], self.paper_gammas[0], 1)   #+self.brwr.Ks[0],
        #        zigzag(D[1], self.paper_gammas[1], 1)   #+self.brwr.Ks[1],
        #        zigzag(D[2], self.paper_gammas[2], 1)   #+self.brwr.Ks[2],
        #    ]) @ self.paper_dyes
        
        #plt.ylim(-0.1, 1.5)
        #plt.plot(developed_dyes[0], 'c--')
        #plt.plot(developed_couplers[0], 'r')
        #plt.plot(developed[0], 'c')
        
        #plt.plot(developed_dyes[1], 'm--')
        #plt.plot(developed_couplers[1], 'g')
        #plt.plot(developed[1], 'm')
        
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

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('film_datasheet')
    parser_test.add_argument('paper_datasheet')
    
    return parser.parse_args()

if __name__ == '__main__':
    opts = arguments()
    film = FilmProfile(opts.film_datasheet, mode31=True)
    paper = FilmProfile(opts.paper_datasheet, mode31=True)
    hanson = Hanson(film, paper)
    hanson.compute_couplers()
    ##for x in np.linspace(0, 1, 10):
    ##    print(hanson.develop(colors.srgb_to_xyz(colors.color(x, x, x))))
    ##plt.plot(hanson.paper_dyes.transpose(), 'k')
    ##plt.plot(hanson.film_dyes.transpose(), 'k--')
    #plt.show()
    hanson.develop_img_srgb('pics/medium21.jpg')
    #print(hanson.to_json())
    #print(hanson.refl_gen.to_json())

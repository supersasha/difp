import argparse

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from profile import FilmProfile
import data
import illum
#import phasetwo
from utils import vectorize
import math

def arguments():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_profile = subparsers.add_parser('profile')
    parser_profile.add_argument('datasheet')
    parser_profile.add_argument('status')
    
    return parser.parse_args()

def sup_idx(idx):
    if idx == 0:
        return [1, 2]
    elif idx == 1:
        return [0, 2]
    else:
        return [0, 1]

def influx(idx, status):
    return data.ILLUM_A @ (10 ** status[idx])

def outflux(idx, status, dye, q, k, y):
    sidx = sup_idx(idx)
    s = q * (k * dye[idx] + (1-k) * (y[0] * dye[sidx[0]] + y[1] * dye[sidx[1]]))
    return data.ILLUM_A @ (10 ** (status[idx] - s))

def density(idx, status, dye, q, k, y):
    ifl = influx(idx, status)
    ofl = outflux(idx, status, dye, q, k, y)
    return np.log10(ifl) - np.log10(ofl)

def couplers(datasheet):
    p = datasheet.profile
    xs = np.arange(data.SPECTRE_SIZE) * 5 + 380
    [red, green, blue] = [datasheet.dye(i) for i in range(3)]
    def solve(basic, coupler1, coupler2):
        def f(x):
            vs = [1.0 / 10**(k*basic + (1-k)*(x[0]*coupler1 + x[1]*coupler2))
                    for k in np.linspace(0, 1, 20)]
            return np.mean(np.std(np.vstack(vs), 0))
        bounds = [(0, 1)] * 2
        return opt.dual_annealing(f, bounds, maxiter=1000)
    reds = solve(red, green, blue)
    greens = solve(green, red, blue)
    blues = solve(blue, red, green)
    plt.subplot(221)
    for k in np.linspace(0, 1, 10):
        plt.plot(xs, 1.0 / 10**(k*red + (1-k)*(reds.x[0]*green + reds.x[1]*blue)), 'r')
    plt.subplot(222)
    for k in np.linspace(0, 1, 10):
        plt.plot(xs, 1.0 / 10**(k*green + (1-k)*(greens.x[0]*red + greens.x[1]*blue)), 'g')
    plt.subplot(223)
    for k in np.linspace(0, 1, 10):
        plt.plot(xs, 1.0 / 10**(k*blue + (1-k)*(blues.x[0]*red + blues.x[1]*green)), 'b')
    plt.show()
    return np.vstack([reds.x, greens.x, blues.x])

def status_by_name(n):
    if n == 'a' or n == 'A':
        return data.STATUS_A
    elif n == 'm' or n == 'M':
        return data.STATUS_M

def profile_cmd(opts):
    datasheet = FilmProfile(opts.datasheet)
    status = status_by_name(opts.status)
    ys = couplers(datasheet)
    print('ys: ', ys)
    dye = datasheet.dye()
    @vectorize()
    def min_curve(idx):
        def f(x):
            j = datasheet.curve(idx, x)
            return j
        bounds = [datasheet.curve_arg_range(idx)]
        res = opt.dual_annealing(f, bounds, maxiter=1000)
        print(res)
        return res.fun
    mc = min_curve([0, 1, 2])
    print('mc: ', mc)
    q = [3.0, 3.0, 3.0]
    d0 = np.array([density(0, status, dye, q[0], 0, ys[0]),
                   density(1, status, dye, q[1], 0, ys[1]),
                   density(2, status, dye, q[2], 0, ys[2])])
    print('d0: ', d0)
    
    ks = np.linspace(0, 1, 100)
    ds = [[], [], []]
    for k in ks:
        for idx in [0, 1, 2]:
            ds[idx].append(density(idx, status, dye, q[idx], k, ys[idx]))
    print('ks: ', ks)
    print('ds: ', ds)

    plt.subplot(111)
    plt.plot(ks, ds[0], 'r')
    plt.plot(ks, ds[1], 'g')
    plt.plot(ks, ds[2], 'b')
    plt.title('ds')
    plt.show()

    print(d0)

    #@vectorize()
    #def solve_q(idx):
    #    #key = datasheet.key_by_index(idx)
    #    def f(q):
    #        #j = datasheet.curve(idx, 100) - density(idx, status, dye, q, 1, ys[idx])
    #        j = mc[idx] - density(idx, status, dye, q, 0, ys[idx])
    #        return j*j
    #    bounds = [(0, 200)]
    #    return opt.dual_annealing(f, bounds, maxiter=1000).x
    #
    #q = solve_q([0, 1, 2])
    print(q)
    @vectorize(exc=[0])
    def solve_k(idx, x):
        def f(k):
            j = datasheet.curve(idx, x)+1*(-mc[idx]+d0[idx]) - density(idx, status, dye, q[idx], k, ys[idx])
            return j*j
        bounds = [(0, 1)]
        res = opt.dual_annealing(f, bounds, maxiter=100)
        print(res.fun)
        return res.x

    profile = datasheet.clone().profile

    for idx in range(3):
        key = datasheet.key_by_index(idx)
        x0, x1 = datasheet.curve_arg_range(idx)
        xs = np.linspace(x0, x1, 100)
        k = solve_k(idx, xs)
        print(k)
        nodes = [{"x": x, "y": y} for (x, y) in zip(xs, k)]
        profile[key]['curve']['nodes'] = nodes
        profile[key]['theta'] = 0
        profile[key]['amp'] = 0

    new_profile = FilmProfile(profile)
    new_profile.profile['red']['couplers'] = [q[0], ys[0, 0], ys[0, 1]]
    new_profile.profile['green']['couplers'] = [ys[1, 0], q[1], ys[1, 1]]
    new_profile.profile['blue']['couplers'] = [ys[2, 0], ys[2, 1], q[2]]
    print(new_profile)
    show(datasheet, 'Datasheet')
    show(new_profile, 'New profile')
    #compare(datasheet, new_profile)
    check(new_profile, status, 'Check new profile')

def check(profile, status, title='check'):
    plt.subplot(111)
    for idx in range(3):
        key = profile.key_by_index(idx)
        x0, x1 = profile.curve_arg_range(idx)
        q = profile.profile[key]['couplers'][idx]
        si = sup_idx(idx)
        ys = [ profile.profile[key]['couplers'][si[0]],
               profile.profile[key]['couplers'][si[1]]]
        xs = np.linspace(x0, x1, 100)
        ks = profile.curve(idx, xs)
        ds = []
        for x in list(xs):
            k = profile.curve(idx, x)
            d = density(idx, status, profile.dye(), q, k, ys)
            ds.append(d)
            
        #def density(idx, status, dye, q, k, y):
        plt.plot(xs, ds, key[0])
    plt.title(title)
    plt.show()


def show(profile, title='show'):
    #plt.subplot(121)
    plt.subplot(111)
    x0, x1 = profile.curve_arg_range(0)
    xs = np.linspace(x0, x1, 100)
    plt.plot(xs, profile.curve(0, xs), 'r')
    plt.plot(xs, profile.curve(1, xs), 'g')
    plt.plot(xs, profile.curve(2, xs), 'b')
    
    #xs = np.linspace(380, 700, 65);
    #plt.subplot(222)
    #plt.plot(xs, profile.sense(0), 'r')
    #plt.plot(xs, profile.sense(1), 'g')
    #plt.plot(xs, profile.sense(2), 'b')

    #plt.subplot(224)
    #plt.plot(xs, profile.dye(0), 'r')
    #plt.plot(xs, profile.dye(1), 'g')
    #plt.plot(xs, profile.dye(2), 'b')
    plt.title(title)
    plt.show()

def compare(old, new):
    plt.subplot(121)
    x0, x1 = old.curve_arg_range(0)
    xs = np.linspace(x0, x1, 100)
    plt.plot(xs, old.curve(0, xs), 'r')
    plt.plot(xs, old.curve(1, xs), 'g')
    plt.plot(xs, old.curve(2, xs), 'b')
    plt.plot(xs, new.curve(0, xs), 'c')
    plt.plot(xs, new.curve(1, xs), 'm')
    plt.plot(xs, new.curve(2, xs), 'y')

    xs = np.linspace(380, 700, 65);
    plt.subplot(222)
    plt.plot(xs, old.sense(0), 'r')
    plt.plot(xs, old.sense(1), 'g')
    plt.plot(xs, old.sense(2), 'b')
    plt.plot(xs, new.sense(0), 'c')
    plt.plot(xs, new.sense(1), 'm')
    plt.plot(xs, new.sense(2), 'y')


    plt.subplot(224)
    plt.plot(xs, old.dye(0), 'r')
    plt.plot(xs, old.dye(1), 'g')
    plt.plot(xs, old.dye(2), 'b')
    plt.plot(xs, new.dye(0), 'c')
    plt.plot(xs, new.dye(1), 'm')
    plt.plot(xs, new.dye(2), 'y')
    
    plt.show()

def main():
    opts = arguments()
    if opts.command == 'profile':
        profile_cmd(opts)

if __name__ == '__main__':
    main()


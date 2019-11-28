from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.decomposition import NMF
import numpy as np

from utils import print_mtx_py
import sys
import matplotlib.pyplot as plt
import math
import data

MAX_ITER = 100

class Node:
    def __init__(self, d, xy):
        self.d = d
        self.xy = xy
        self.bounds = [[1.0, 0.0], [1.0, 0.0]]
        
        for z in xy:
            for i in range(2):
                if self.bounds[i][0] > z[i]:
                    self.bounds[i][0] = z[i]
                if self.bounds[i][1] < z[i]:
                    self.bounds[i][1] = z[i]

        #print('Bounds:', self.bounds)

        model = NMF(n_components=3, max_iter=MAX_ITER, tol=1e-20)
        model.fit_transform(d.transpose())
        self.e = model.reconstruction_err_ ** 2
        self.axis = None
        self.k = None
        self.less = None
        self.more = None

    def is_leaf(self):
        return self.axis is None

    def error(self):
        if self.is_leaf():
            return self.e
        else:
            return self.less.error() + self.more.error()
    
    def step(self):
        if self.is_leaf():
            self.own_step()
        else:
            if self.less.error() < self.more.error():
                self.more.step()
            else:
                self.less.step()

    def own_step(self):
        e0, k0 = self.axis_step(0)
        e1, k1 = self.axis_step(1)
        if e0 < e1:
            self.axis = 0
            self.k = k0
        else:
            self.axis = 1
            self.k = k1

        #print('axis:', 'x' if self.axis == 0 else 'y', 'k:', self.k)

        d_less = []
        d_more = []
        xy_less = []
        xy_more = []
        for i in range(self.d.shape[0]):
            if self.xy[i][self.axis] < self.k:
                d_less.append(self.d[i])
                xy_less.append(self.xy[i])
            else:
                d_more.append(self.d[i])
                xy_more.append(self.xy[i])
        self.less = Node(np.vstack(d_less), np.vstack(xy_less))
        self.more = Node(np.vstack(d_more), np.vstack(xy_more))

    def axis_step(self, axis):
        def f(x):
            k = x[0]
            d1 = []
            d2 = []
            e1 = 0
            e2 = 0
            for i in range(self.d.shape[0]):
                if self.xy[i][axis] < k:
                    d1.append(self.d[i])
                else:
                    d2.append(self.d[i])
            if len(d1) > 0:
                d1 = np.vstack(d1)
                model1 = NMF(n_components=3, max_iter=MAX_ITER, tol=1e-20)
                model1.fit_transform(d1.transpose())
                e1 = model1.reconstruction_err_

            if len(d2) > 0: 
                d2 = np.vstack(d2)
                model2 = NMF(n_components=3, max_iter=MAX_ITER, tol=1e-20)
                model2.fit_transform(d2.transpose())
                e2 = model2.reconstruction_err_

            e = e1*e1 + e2*e2

            #print('---------------------------')
            #print('k:', k)
            #print('norm:', e, e1, e2)
            #print('#d1:', len(d1))
            #print('#d2:', len(d2))
            return e

        bounds = [self.bounds[axis]] #[(0, 1)]
        r = opt.dual_annealing(f, bounds, maxiter=100)
        return r.fun, r.x[0]

    def axis_name(self):
        if self.axis == 0:
            return 'x'
        else:
            return 'y'

    def print(self, lvl=0):
        indent = '    ' * lvl
        if self.is_leaf():
            print(f'{indent}n: {self.d.shape[0]}')
        else:
            print(f'{indent}axis: {self.axis_name()}, k: {self.k}')
            self.less.print(lvl=lvl+1)
            self.more.print(lvl=lvl+1)


def main():
    mat = loadmat('TotalRefs_IndividualSpectra.mat')
    d = mat['TotalRefs_IndividualSpectra']
    #d = d[:20000]
    t = d.dot(data.A_1931_64_400_700_10nm)
    tv = t[:, 0] + t[:, 1] + t[:, 2]
    xy = (t.transpose() / tv).transpose()[:,0:2]

    n = Node(d, xy)
    print(f'Initial error: {n.error()}')

    #print(t.shape, tv.shape, xy.shape)
    plt.scatter(xy[:, 0], xy[:, 1], s=1)
    plt.axis([0, 1, 0, 1])
    plt.show()

    for i in range(8):
        n.step()
        print(f'step: {i}, error: {n.error()}')

    print()
    n.print()

#    def f(x):
#        nu = x[0]
#        d1 = []
#        d2 = []
#        for i in range(d.shape[0]):
#            if xy[i][0] < nu:
#                d1.append(d[i])
#            else:
#                d2.append(d[i])
#       
#        e1 = 0
#        e2 = 0
#
#        if len(d1) > 0:
#            d1 = np.vstack(d1)
#            model1 = NMF(n_components=3, max_iter=100, tol=1e-20)
#            model1.fit_transform(d1.transpose())
#            e1 = model1.reconstruction_err_
#
#        if len(d2) > 0: 
#            d2 = np.vstack(d2)
#            model2 = NMF(n_components=3, max_iter=100, tol=1e-20)
#            model2.fit_transform(d2.transpose())
#            e2 = model2.reconstruction_err_
#
#        e = math.sqrt(e1*e1 + e2*e2)
#
#        print('---------------------------')
#        print('nu:', nu)
#        print('norm:', e, e1, e2)
#        print('#d1:', len(d1))
#        print('#d2:', len(d2))
#        return e
#
#    bounds = [(0, 1)]
#    r = opt.dual_annealing(f, bounds, maxiter=100)
#    print('r:', r)
    
    #model = NMF(n_components=3, max_iter=100, tol=1e-20)
    #U = model.fit_transform(d.transpose())
    #print(math.sqrt(model.reconstruction_err_) / d.shape[0], file=sys.stderr)
    #print('import numpy as np\n')
    #print('SPECTRAL_BASIS = ', end='')
    #print_mtx_py(U)
    #plt.plot(U)
    #plt.show()

if __name__ == '__main__':
    main()


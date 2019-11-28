from munsell_data import MUNSELL_DATA
from sklearn.decomposition import NMF
from utils import print_mtx_py
import sys

def main():
    model = NMF(n_components=3, init='random', max_iter=10000, tol=1e-20)
    U = model.fit_transform(MUNSELL_DATA.transpose())
    print(model.reconstruction_err_, file=sys.stderr)
    print('import numpy as np\n')
    print('MUNSELL_BASIS = ', end='')
    print_mtx_py(U)

if __name__ == '__main__':
    main()

from functools import wraps
import numpy as np
from scipy.interpolate import CubicSpline

def unzip(es):
    xs = np.array([x[0] for x in es])
    ys = np.array([x[1] for x in es])
    return (xs, ys)

def vectorize(otypes=None, sig=None, exc=None):
    """Numpy vectorization wrapper that works with instance methods."""
    def decorator(fn):
        vectorized = np.vectorize(fn, otypes=otypes, signature=sig, excluded=exc)
        @wraps(fn)
        def wrapper(*args):
            return vectorized(*args)
        return wrapper
    return decorator

def print_mtx_py(mtx, indent_level=0, indent_size=4):
    (rows, cols) = mtx.shape
    print('np.array([', end='')
    first_row = True
    for i in range(rows):
        if first_row:
            first_row = False
        else:
            print(',', end='')
        print('\n' + ' '*(indent_size*(indent_level + 1)) + '[ ', end='')
        first_col = True
        for j in range(cols):
            if first_col:
                first_col = False
            else:
                print(', ', end='')
            print(mtx[i, j], end='')
        print(' ]', end='')
    print('\n' + ' '*(indent_level*indent_size) + '])')

#@vectorize()
def to_400_700_10nm(vec_380_700_5nm):
    xs = np.linspace(380, 700, 65)
    xsp = np.linspace(400, 700, 31)
    sp = CubicSpline(xs, vec_380_700_5nm, bc_type='natural')
    return sp(xsp)

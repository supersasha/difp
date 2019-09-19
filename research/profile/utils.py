from functools import wraps
import numpy as np

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



import numpy as np
import numpy.typing as npt
from numba import jit, prange

Float = np.float64
Vector = npt.NDArray[Float]

@jit(nopython=True, nogil=True)
def calc_and(cc: int, stat: Vector, ct: Vector, fv: Vector) -> Vector:
    p: Vector = np.zeros(cc, dtype=Float)
    for ci in prange(cc):
        # p[ci] = np.multiply.reduce(1 - stat[ci] / ct[ci] * (1 - fv))
        p[ci] = np.prod(1 - stat[ci] / ct[ci] * (1 - fv))
    return p

@jit(nopython=True, nogil=True)
def calc_or(cc: int, stat: Vector, ft: Vector, fv: Vector) -> Vector:
    p: Vector = np.zeros(cc, dtype=Float)
    for ci in prange(cc):
        # p[ci] = np.frompyfunc(lambda x, y: (x + y) - (x * y), 2, 1).reduce(self.stat[ci] / self.ft * fv)
        # p[ci] = 1 - np.multiply.reduce(1 - self.stat[ci] / self.ft * fv)
        p[ci] = 1 - np.prod(1 - stat[ci] / ft * fv)
    return p

class LayerOr():
    def __init__(self, classes_count: int, features_count: int, tory: float = 0.0001):
        self.cc = classes_count
        self.fc = features_count
        self.ft = np.full(features_count, tory, dtype=Float) # totals
        self.stat = np.zeros((classes_count, features_count), dtype=Float)

    def feed(self, ci: int, fv: Vector):
        self.stat[ci] += fv
        self.ft += fv
    
    def calc(self, fv: Vector) -> Vector:
        return calc_or(self.cc, self.stat, self.ft, fv)
    
class LayerAnd():
    def __init__(self, classes_count: int, features_count: int, tory: float = 0.0001):
        self.cc = classes_count
        self.fc = features_count
        self.ct = np.full(classes_count, tory, dtype=Float) # totals
        self.stat = np.zeros((classes_count, features_count), dtype=Float)

    def feed(self, ci: int, fv: Vector):
        self.ct[ci] += 1
        self.stat[ci] += fv
    
    def calc(self, fv: Vector) -> Vector:
        return calc_and(self.cc, self.stat, self.ct, fv)
    
def compl(x):
    x = np.array(x, dtype=Float)
    return np.append(x, 1-x)

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    x = [
        (0, [0.0, 1.0, 1.0]),
        (0, [0.0, 1.0, 1.0]),
        (0, [0.0, 1.0, 1.0]),
        (0, [0.0, 1.0, 1.0]),
        (0, [0.0, 1.0, 1.0]),
        (0, [0.0, 1.0, 1.0]),
        (0, [0.0, 1.0, 1.0]),
        (0, [0.0, 1.0, 1.0]),
        (0, [0.0, 1.0, 1.0]),
        (0, [0.0, 1.0, 1.0]),
        (1, [1.0, 1.0, 1.0]),
    ]

    c0 = LayerOr(2, 3)
    for r in x:
        c0.feed(r[0], r[1])

    p = c0.calc(np.array([0.0, 1.0, 1.0]))
    print(p.argmax(), p)

    c1 = LayerAnd(2, 3*2)
    for r in x:
        c1.feed(r[0], compl(r[1]))

    p = c1.calc(compl([0.0, 1.0, 1.0]))
    print(p.argmax(), p)
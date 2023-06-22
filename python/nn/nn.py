import numpy as np
import numpy.typing as npt

Float = np.float64
Vector = npt.NDArray[Float]

class LayerOr():
    def __init__(self, classes_count: int, features_count: int, tory: float = 0.0001):
        self.ft = np.full(features_count, tory, dtype=Float) # totals
        self.stat = np.zeros((classes_count, features_count), dtype=Float)

    def feed(self, ci: int, fv: Vector):
        self.stat[ci] += fv
        self.ft += fv
    
    def calc(self, fv: Vector) -> Vector:
        return 1 - np.prod(1 - self.stat / self.ft * fv, axis=1)

class LayerAnd():
    def __init__(self, classes_count: int, features_count: int, tory: float = 0.0001):
        self.ct = np.full((classes_count, 1), tory, dtype=Float) # totals
        self.stat = np.zeros((classes_count, features_count), dtype=Float)

    def feed(self, ci: int, fv: Vector):
        self.ct[ci] += 1
        self.stat[ci] += fv
    
    def calc(self, fv: Vector) -> Vector:
        w = self.stat / self.ct
        return np.prod((1 - w*(1-fv)) * (1 - (1-w)*fv), axis=1)

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
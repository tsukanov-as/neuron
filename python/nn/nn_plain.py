import numpy as np
import numpy.typing as npt

Float = np.float64
Vector = npt.NDArray[Float]

def NOT(x):
	return 1 - x

def AND(x, y):
	return x * y

def OR(x, y):
	return (x + y) - (x * y)

def OR2(x, y):
	return NOT(AND(NOT(x), NOT(y)))

def XOR(x, y):
    return OR(AND(NOT(x), y), AND(x, NOT(y)))

def XNOR(x, y):
	return NOT(XOR(x, y))

def XNOR2(x, y):
	return AND(OR(NOT(x), y), OR(x, NOT(y)))

def XNOR3(x, y):
	return AND(OR2(NOT(x), y), OR2(x, NOT(y)))

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
        p = np.zeros(self.cc, dtype=Float)
        for ci in range(self.cc):
            cp = 0
            cf = self.stat[ci]
            for fi in range(self.fc):
                fp = AND(cf[fi] / self.ft[fi], fv[fi])
                cp = OR(cp, fp)
            p[ci] = cp
        return p

class LayerAnd():
    def __init__(self, classes_count: int, features_count: int, tory: float = 0.0001):
        self.cc = classes_count
        self.fc = features_count
        self.ct = np.full((classes_count, 1), tory, dtype=Float) # totals
        self.stat = np.zeros((classes_count, features_count), dtype=Float)

    def feed(self, ci: int, fv: Vector):
        self.ct[ci] += 1
        self.stat[ci] += fv
      
    def calc(self, fv: Vector) -> Vector:
        p = np.zeros(self.cc, dtype=Float)
        for ci in range(self.cc):
            cp = 1
            cf = self.stat[ci]
            for fi in range(self.fc):
                fp = XNOR(cf[fi] / self.ct[ci], fv[fi])
                cp = AND(cp, fp)
            p[ci] = cp
        return p

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

    p = c0.calc([0.0, 1.0, 1.0])
    print(p.argmax(), p)

    c1 = LayerAnd(2, 3)
    for r in x:
        c1.feed(r[0], r[1])

    p = c1.calc([0.0, 1.0, 1.0])
    print(p.argmax(), p)
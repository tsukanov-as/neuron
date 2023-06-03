import torch

Float = torch.float64
Vector = torch.Tensor

class LayerOr():
    def __init__(self, classes_count: int, features_count: int, tory: float = 0.0001):
        self.ft = torch.full((features_count,), tory, dtype=Float) # totals
        self.stat = torch.zeros((classes_count, features_count), dtype=Float)

    def feed(self, ci: int, fv: Vector):
        self.stat[ci] += fv
        self.ft += fv
    
    def calc(self, fv: Vector) -> Vector:
        return 1 - torch.prod(1 - self.stat / self.ft * fv, dim=1)
    
class LayerAnd():
    def __init__(self, classes_count: int, features_count: int, tory: float = 0.0001):
        self.ct = torch.full((classes_count, 1), tory, dtype=Float) # totals
        self.stat = torch.zeros((classes_count, features_count), dtype=Float)

    def feed(self, ci: int, fv: Vector):
        self.ct[ci] += 1
        self.stat[ci] += fv
    
    def calc(self, fv: Vector) -> Vector:
        return torch.prod(1 - self.stat / self.ct * (1 - fv), dim=1)
    
def compl(x):
    x = torch.tensor(x, dtype=Float)
    return torch.cat([x, 1-x])

if __name__ == "__main__":
    torch.set_printoptions(precision=3, sci_mode=False)

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
        c0.feed(r[0], torch.tensor(r[1]))

    p = c0.calc(torch.tensor([0.0, 1.0, 1.0]))
    print(p.argmax(), p)

    c1 = LayerAnd(2, 3*2)
    for r in x:
        c1.feed(r[0], compl(r[1]))

    p = c1.calc(compl([0.0, 1.0, 1.0]))
    print(p.argmax(), p)
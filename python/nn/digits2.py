from sklearn.datasets import load_digits
digits = load_digits()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.05, random_state=42) # type: ignore

import numpy as np
from nn import LayerOr, LayerAnd

x_train = x_train.astype(float) / 16
x_test = x_test.astype(float) / 16

x_train = np.append(x_train, 1-x_train, axis=1)
x_test = np.append(x_test, 1-x_test, axis=1)

K = 15
N = 10*K

la = LayerAnd(N, 8*8*2, 100)

for i in range(N):
    la.feed(i, np.random.rand(8*8*2))

for _ in range(200):
    for i, v in enumerate(x_train):
        base = y_train[i]*K
        p = la.calc(v)[base:base+K]
        la.feed(base + p.argmax(), v)
        # la.feed(la.calc(v).argmax(), v)

lo = LayerOr(10, N, 0)
for i, v in enumerate(x_train):
    p = la.calc(v)
    lo.feed(y_train[i], p)

total = 0
for i, v in enumerate(x_test):
    p = lo.calc(la.calc(v))
    if p.argmax() == y_test[i]:
        total += 1

print("LayerAnd -> LayerOr = {:.2f}%".format(total / len(x_test) * 100))
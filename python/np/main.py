import numpy as np

v = np.array([0.5, 0.5, 0.1])
print("vec:", v)

# вычисление NOT(x) = 1-x
print("not:", 1-v)

# вычисление AND(x, y) = x * y
print("and:", np.prod(v))

# вычисление OR(x, y) = (x + y) - (x * y)
print("or:", np.frompyfunc(lambda x, y: (x + y) - (x * y), 2, 1).reduce(v))

# так как OR(x, y) = NOT(AND(NOT(x), NOT(y)))
print("or:", 1-np.prod(1-v))

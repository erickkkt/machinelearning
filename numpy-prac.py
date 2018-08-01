from __future__ import print_function
import numpy as np
a = np.arange(15).reshape(3,5)
print(a)
print(a.shape)
print(a.ndim)
print(a.size)
b = np.array([1,2,3,4,5,6])
print(b)
c = np.array([[1,2],[3,4]], dtype=complex)
print(c)

d = np.zeros((3,4))
print (d)

e = np.ones((2,3,4), dtype=np.int16)
print(e)
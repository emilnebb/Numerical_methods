import numpy as np
import matplotlib.pyplot as plt
import scipy as sip
import sympy as sp

a = np.array([1,2,3])
b = np.array([4,5,6])

A = np.array([[1,1,2],[2,3,3],[4,4,5]])
B = np.array([[2,4,6], [8,10,12],[14,16,18]])

print(a+b)
print(sum(a*b))
print(A*B)
print(np.transpose(A))
print(np.invert(A))

x = np.linalg.solve(A,b)
print(x)





import numpy as np
import matplotlib.pyplot as plt
import scipy as sip
import sympy as sp
import scipy.special as sss

#defining the function y'=f(x,y):
def f(x,y):
    return 1-3*x + y + x**2 +x*y

#Newton's solution:
def Newtons_sol(x):  
    return x - x**2 + (x**3)/3 - (x**4)/6 + (x**5)/30 - (x**6)/45

#Analytical solution:
def a_sol(x):
    return (3*np.sqrt(2*np.pi*np.e)*np.exp(x*(1+x/2))*(sss.erf((np.sqrt(2)/2)*(1+x))- sss.erf(np.sqrt(2)/2)) + 
    4*(1- np.exp(x*(1+x/2)))-x)

"""General Euler's method, input-parameters:
- x0: x-start
- x1: x-end
- h: step-size
- y0: initial condition y(x0) = y0
- f: f=y'(x)
"""

def eulers(x0,x1, h, y0, f):
    
    x= np.arange(x0,x1+h,h)
    y = np.zeros(np.size(x),float)

    for i in range(np.size(x)):
        y[i] = y0
        y0 = y0 + h*f(x0,y0) #the Euler's step of computing next value of y
        x0 = x0 + h
    
    return x,y

x0 = 0.0
x1 = 1.5
h = 0.1
y0 = 0
x,y = eulers(x0,x1,h, y0, f)

plt.plot(x,y, linestyle = '-.',label = "Eulers")
plt.plot(x, Newtons_sol(x), label = "Newton's solution")
plt.plot(x, a_sol(x), label = "Analytical solution")
plt.legend()
plt.grid()
plt.show()




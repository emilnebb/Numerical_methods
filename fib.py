import numpy as np
import matplotlib.pyplot as plt
import scipy as sip
import sympy as sp
import pylab as pl

def fib(n):
    if n < 0:
        print("Invalid input argument")
        return
    f1 = 0
    f2 = 1
    
    for i in range(n):
      print(f1)
      next = f1 + f2
      f1 = f2
      f2 = next

fib(20)

def fibplot(n):
    if n < 0:
        print("Invalid input argument")
        return
    f1 = 0
    f2 = 1
    serie = np.zeros(n, int)
    
    for i in range(n):
      #print(f1)
      serie[i] = f1
      next = f1 + f2
      f1 = f2
      f2 = next 

    print(np.arange(0,n))
    print(serie)
    plt.plot(np.arange(0,n),serie)
    plt.xlabel("x-verdi")
    plt.yscale('log')
    plt.ylabel("y-verdi")
    plt.title("Fibonacci")
    plt.legend()
    plt.grid()
    plt.show()

fibplot(30)

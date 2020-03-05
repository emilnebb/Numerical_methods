import numpy as np
import matplotlib.pyplot as plt
import scipy as sip

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def tempBeam(Tleft, Tright, T0, L, deltaT, deltaX, endTime):

    D = deltaT/deltaX**2

    #Checking if the FTCS will be stable or not:
    print("Checking stability")
    print("------------------")

    if (D< 0.5):
        print("Stability check okey")
    else:
        print("WARNING --> unstable for given parameters")
        return

    T = np.int(endTime/deltaT)  #seting number of iterations
    X = np.int(L/deltaX + 1)  #seting number of iterations along the beam
    
    TempPrev = np.zeros(X, float) #initial condition is that all the temperatures along the beam is zero
    TempNext = np.zeros(X, float)

    for j in range(0,T):
        for i in range(0, X):
            if (i == 0):
                TempNext[i] = 100   #left boundary condition
            elif (i == X-1):
                TempNext[i] = 0     #right boundary condition
            elif (i == 1):
                TempNext[i] = D*(TempPrev[i+1] + 100) + (1 - 2*D)*TempPrev[i]
            elif (i == X-1):
                TempNext[i] = D*(TempPrev[i-1]) + (1 - 2*D)*TempPrev[i]
            else :
                TempNext[i] = D*(TempPrev[i-1] + TempPrev[i+1]) + (1 - 2*D)*TempPrev[i]
        TempPrev = TempNext
    
    return TempNext

#--End-of-function

Temp1 = tempBeam(100, 0, 0, 1, 10**(-5), 0.01, 0.001 )
Temp2 = tempBeam(100, 0, 0, 1, 10**(-5), 0.01, 0.025 )
Temp3 = tempBeam(100, 0, 0, 1, 10**(-5), 0.01, 0.4 )

x = np.linspace(0, 1, 101)

plt.plot(x, Temp1, label = "t=0.001")
plt.plot(x, Temp2, label = "t=0.025")
plt.plot(x, Temp3, label = "t=0.4")
plt.legend()
plt.ylabel("Temperature")
plt.xlabel("x-position")
plt.grid()
plt.show()
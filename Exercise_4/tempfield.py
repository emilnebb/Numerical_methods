import numpy as np
import matplotlib.pyplot as plt
import scipy as sip

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def tempfield(Tleft, Ttop, l, N):

    A = np.zeros((N*N,N*N), int)
    b = np.zeros((N*N), int)

    for j in range (0,N):
        for i in range (1,N+1):
            A[j*N + i-1,j*N + i-1] = -4                 #setting the diagonals to -4

            if (i>1):
                A[j*N + i-1, j*N + i -2] = 1
            if (i<N):
                A[j*N + i-1, j*N + i] = 1  
            if (j>0):
                A[j*N + i-1, j*N + i -(N+1)] = 1
            if (j<N-1):
                A[j*N + i-1, j*N + i +(N-1)] = 1

            if (j == 0):                    #if we are at the bottum of the plate, we need to use ghost nodes
                A[j*N + i-1, i + (N-1)] = 2
            
            if (i == N):
                A[j*N+ i-1 , j*N + i -2] = 2
            
            if ((i == 1)and(j == N-1)):     #if we are at the node next to the top left corner
                b[j*N + i-1] = -(Tleft + Ttop)
                A[j*N + i-1,j*N + i] = 1
                A[j*N + i-1,j*N + i-1 - N] = 1
            elif ((i == 1)and (j == 0 )):
                b[j*N + i-1] = -Tleft
                A[j*N + i-1,j*N + i] = 1
                A[j*N + i-1,j*N + i-1 + N] = 2
            elif (i == 1):                  #if we are next to the left boundary
                b[j*N + i-1] = -Tleft
                A[j*N + i-1,j*N + i] = 1
                A[j*N + i-1,j*N + i-1 + N] = 1
                A[j*N + i-1,j*N + i-1 - N] = 1
            elif ((j == N-1)and(i == N)):
                b[j*N + i-1] = -Ttop            #if we are next to the top boundary
                A[j*N + i-1,j*N + i-2] = 2
                A[j*N + i-1,j*N + i-1 -N] = 1
            elif (j == N-1):
                b[j*N + i-1] = -Ttop 
                A[j*N + i-1,j*N + i-2] = 1
                A[j*N + i-1,j*N + i] = 1
                A[j*N + i-1,j*N + i-1 -N] = 1    
    return A,b
            
#--End-of-function

def plotSurfaceNeumannDirichlet(Temp, Ttop, Tleft, l, N, nxTicks=4, nyTicks=4):
    
    """ Surface plot of the stationary temperature in quadratic beam cross-section.
        Note that the components of T has to be started in the
        lower left part of the grid with increasing indexes in the
        x-direction first.
    
    
         Args:
             Temp(array):  the unknown temperatures, i.e. [T_1 .... T_(NxN)]
             Ttop(float):  temperature at the top boundary
             Tleft(float): temperature at the left boundary
             l(float):     height/width of the sides
             N(int):       number of nodes with unknown temperature in x/y direction
             nxTicks(int): number of ticks on x-label (default=4)
             nyTicks(int): number of ticks on y-label (default=4)
    """
    x = np.linspace(0, l, N + 1)
    y = np.linspace(0, l, N + 1)
    
    X, Y = np.meshgrid(x, y)
    
    T = np.zeros_like(X)
    
    T[-1,:] = Ttop
    T[:,0] = Tleft
    k = 1
    for j in range(N):
        T[j,1:] = Temp[N*(k-1):N*k]
        k+=1

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, T, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_zlim(0, Ttop)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('T [$^o$C]')
    
    xticks=np.linspace(0.0, l, nxTicks+1)
    ax.set_xticks(xticks)
    
    yticks=np.linspace(0.0, l, nyTicks+1)
    ax.set_yticks(yticks)
    plt.show()

#-End-of-function

A,b = tempfield(50,100,1,50)
A_invers = np.linalg.inv(A)

Temptemp = A_invers * b
Temp = np.zeros(np.size(b), float)
for i in range(np.size(Temp)):
    Temp[i] = np.sum(Temptemp[i,:])

#print(Temp)
plotSurfaceNeumannDirichlet(Temp, 100, 50, 1, 50)

#ploting a 50x50 temperature mesh



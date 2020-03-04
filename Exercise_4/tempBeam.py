import numpy as np
import matplotlib.pyplot as plt
import scipy as sip

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def tempfield(Tleft, Tright, T0, l, N, timeEnd):

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
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy as sip

#Declaring constants
l = 1
t_start = 0
t_end =0.4
a_constant = 1
c = 1
N = 100
delta_x = 1/N
delta_t = delta_x*c/a       #delta t found from the CFL-condition
x = np.linspace(0, l, N + 1)

#t = np.linspace(t_start,t_end, N*c*a + 1)

def aNonLinear(u):
    return 0.9 + 0.1*u

def del

def D(a, delta_t, delta_x):
    return a*delta_t/delta_x

#initial advection equation, u0 when time is zero, is given by:
def u0(x):
    u0 = np.zeros(np.size(x), float)
    for i in range(np.size(x)):
        if (x[i] <0.2):
            u0[i] = (np.sin(np.pi*(x[i]/0.2)))**2
        else:
            u0[i] = 0
    return u0

#Analytical solution is given by:
def u_analytical(u0,x,a,t):
    return u0(x-a*t)

#Computing the numerical solution of the next time step in the advection equation
    #This is equivalent to one iteration through the numerical scheeme
    #The only difference between the LINEAR and NON_LINEAR scheme, is wether a is constant or non-constant
def nextTimeStepU_numerical(u_prev, x, a, D):

    u_next = np.zeros(np.size(x), float)
    for j in range(1,np.size(x)):
        u_next[j] = u_prev[j] - D*(u_prev[j] - u_prev[j-1])
    return u_next
    

time = np.arange(t_start, t_end + delta_t, delta_t)

#Preallocating vectors for analytical solution and numerical solution. Need to keep the previous step to calculate the next one
u_linear0 = u_analytical(u0,x,a_constant,0) 
u_linear1 = np.zeros(np.size(x), float)

u_Analytical = u_analytical(u0,x,a_constant,0) 


#Incrementing through the numerical scheeme
for i in range(1,np.size(time)):
    u_linear1 = nextTimeStepU_numerical(u_linear0,x, a_constant, D)
    u_Analytical = u_analytical(u0,x,a_constant,time[i])
    u_linear0 = u_linear1

#This will plot the result for the last time-step
plt.plot(x, u_Analytical, label = "analytical")
plt.plot(x, u_linear1, label = "Numerical solution")
plt.legend()
plt.ylabel("Deviation")
plt.xlabel("x-position")
plt.grid()
plt.show()
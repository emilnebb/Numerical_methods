import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy as sip

#Declaring constants
l = 1
t_start = 0
t_end = 0.8
a_constant = 1
c = 1
N = 100
global delta_x
delta_x = (1/N)
global delta_t
delta_t = delta_x*c/a_constant       #delta t found from the CFL-condition. Also asuming that we are using a constant delta t during the nonlinear numerical scheme
x = np.linspace(0, l, N + 1)


#t = np.linspace(t_start,t_end, N*c*a + 1)

def aNonLinear(u):
    return 0.9 + 0.1*u

"""def D(a, delta_t, delta_x):
    return a*delta_t/delta_x"""

#initial advection equation, u0 when time is zero, is given by:
def u0(x):
    u0 = np.zeros(np.size(x), float)
    for i in range(np.size(x)):
        if (x[i] <0.2 and x[i]> 0):
            u0[i] = (np.sin(np.pi*(x[i]/0.2)))**2
        else:
            u0[i] = 0
    return u0

#Analytical solution is given by:
def u_analytical(u0,x,a,t):
    return u0(x-a*t)

#Computing the numerical solution of the next time step in the advection equation
    #This is equivalent to one iteration through the numerical scheeme
def linear(u_prev, x, a):

    D = a*delta_t/delta_x
    u_next = np.zeros(np.size(x), dtype=np.float)
    for j in range(1,np.size(x)):
        u_next[j] = u_prev[j] - D*(u_prev[j] - u_prev[j-1])
    return u_next

    #The non-linear numerical scheme, asuming a is a vector with different values. Resulting in D becoming a vector.
def nonLinear(u_prev, x, a):

    D = a*delta_t/delta_x
    u_next = np.zeros(np.size(x), dtype=np.float64)
    for j in range(1,np.size(x)):
        u_next[j] = u_prev[j] - D[j]*(u_prev[j] - u_prev[j-1])
    return u_next

def conservative(u_prev, x):    #The expression of F is: 0.9*u + 0.1*u**2
   
    F = np.zeros(np.size(u_prev), dtype=np.float32)
    for i in range(np.size(u_prev)):
        F[i] = np.float32(0.9*u_prev[i] + 0.1*(np.float32(u_prev[i]**2)))
    F = np.array(F, dtype=np.float32)
    u_next = np.zeros(np.size(x), dtype=np.float64)
    for j in range(1,np.size(x)):
        u_next[j] = u_prev[j] - (delta_t/delta_x)*(np.float64(F[j]- F[j-1]))
    return u_next

#Preallocating vectors for analytical solution and numerical solution. Need to keep the previous step to calculate the next one
time = np.arange(t_start, t_end + delta_t, delta_t)

u_linear0 = u_analytical(u0,x,a_constant,0) 
u_nonlinear0 = u_analytical(u0,x,aNonLinear(u0(x)),0)
u_conservative0 = u_analytical(u0,x,a_constant,0) 

u_Analytical = u_analytical(u0,x,a_constant,0) 


#Incrementing through the numerical scheeme
for i in range(1,np.size(time)):
    u_linear1 = linear(u_linear0,x, a_constant)
    u_linear1[np.size(u_linear1)-1] = u_analytical(u0,x-a_constant*delta_t, a_constant, time[i])[np.size(u_linear1)-1]    #setting right boundary condition
    
    u_nonlinear1 = nonLinear(u_nonlinear0, x, aNonLinear(u_nonlinear0))
    u_nonlinear1[np.size(u_nonlinear1)-1] = u_analytical(u0,x-aNonLinear(u_nonlinear0)*delta_t, a_constant, time[i])[np.size(u_nonlinear1)-1]   #setting right boundary condition
    
    u_conservative1 = conservative(u_conservative0, x)
    #u_Analytical = u_analytical(u0,x,a_constant,time[i])
    u_linear0 = u_linear1
    u_nonlinear0 = u_nonlinear1
    u_conservative0 = u_conservative1

#This will plot the result for the last time-step
plt.plot(x, u_analytical(u0,x,a_constant,0.8), label = "Analytical")
plt.plot(x, u_linear1, linestyle = 'dashed', label = "Numerical linear solution")
plt.plot(x, u_nonlinear1,linestyle = 'dashed', label = "Numerical non-linear solution")
plt.plot(x, u_conservative1, linestyle = 'dashed', label = "Numerical conservative solution")
plt.legend()
plt.title("Solution at time = %s" % t_end)
plt.ylabel("Deviation")
plt.xlabel("x-position")
plt.grid()
plt.show()
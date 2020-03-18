import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy as sip



#Declaring constants
l = 1
t_start = 0
t_end =0.8
a = 1
c = 1
N = 100
delta_x = 1/N
delta_t = delta_x*c/a       #delta t found from the CFL-condition
x = np.linspace(0, l, N + 1)
D = a*delta_t/delta_x
#t = np.linspace(t_start,t_end, N*c*a + 1)

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

def nextTimeStepU_numerical(u_prev, x, a, D):

    u_next = np.zeros(np.size(x), float)
    for j in range(1,np.size(x)):
        u_next[j] = u_prev[j] - D*(u_prev[j] - u_prev[j-1])
    return u_next
    

time = np.arange(t_start, t_end + delta_t, delta_t)

   
u_prev = u_analytical(u0,x,a,0) 
u_Analytical = u_analytical(u0,x,a,0) 
u_next = np.zeros(np.size(x), float)

for i in range(1,np.size(time)):
    u_next = nextTimeStepU_numerical(u_prev,x, a, D)
    u_Analytical = u_analytical(u0,x,a,time[i])
    u_prev = u_next



#print (t)
plt.plot(x, u_analytical(u0,x,a,0.8), label = "analytical")
plt.plot(x, u_next, label = "Numerical solution")
plt.legend()
plt.ylabel("Deviation")
plt.xlabel("x-position")
plt.grid()
plt.show()
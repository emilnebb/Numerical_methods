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





print (x)
#print (t)
plt.plot(x, u_analytical(u0,x,a,0), label = "analytical")
plt.legend()
plt.ylabel("Deviation")
plt.xlabel("x-position")
plt.grid()
plt.show()
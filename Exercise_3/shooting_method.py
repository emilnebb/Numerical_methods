import matplotlib
import numpy as np
import matplotlib.pylab as plt
import scipy as sip

#Setting functions
def rk4(func, z0, time):
    """The Runge-Kutta 4 scheme for solution of systems of ODEs.
    z0 is a vector for the initial conditions,
    the right hand side of the system is represented by func which returns
    a vector with the same size as z0 ."""

    z = np.zeros((np.size(time),np.size(z0)))
    z[0,:] = z0
    zp = np.zeros_like(z0)

    for i, t in enumerate(time[0:-1]):
        dt = time[i+1] - time[i]
        dt2 = dt/2.0
        k1 = np.asarray(func(z[i,:], t))                # predictor step 1
        k2 = np.asarray(func(z[i,:] + k1*dt2, t + dt2)) # predictor step 2
        k3 = np.asarray(func(z[i,:] + k2*dt2, t + dt2)) # predictor step 3
        k4 = np.asarray(func(z[i,:] + k3*dt, t + dt))   # predictor step 4
        z[i+1,:] = z[i,:] + dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4) # Corrector step
    
    return z

def secant(phi0,phi1,s0,s1):
    if (abs(phi1-phi0)>0.0):   
        return    -phi1 *(s1 - s0)/float(phi1 - phi0)
    else:
        return 0.0

def f(y, x):
    yout = np.zeros_like(y)
    yout[:] = [y[1],-2*x*y[0]**2]
    return yout

#Declaring variables
N = 90
x = np.linspace(0, 0.9, N + 1)

y_analytic = 0.5*(np.log(np.abs(x-1))-np.log(np.abs(x+1))) + 2

ystart = 2
yend = 0.5*(np.log(0.1)-np.log(1.9))+2

s = [-0.5, -0.6]

y0 = np.zeros(2)
y0[0] = 2
y0[1] = s[0]

y = rk4(f, y0, x)
phi0 = y[-1,0] - yend
y = y[:,0]

nmax = 10
eps = 0.000001

#Making initial plots
plt.figure()
linestyles = ['0.75', 'r--', 'g--', 'b--', 'y--', 'm--', 'c--']
plt.plot(x,y_analytic)
plt.plot(x,y)
legendList = ['analytic', 'it = 0']

#Iterating for y, and plotting
for i in range(nmax):
    y0[1] = s[1]
    y = rk4(f, y0,x)
    phi1 = y[-1,0] - yend
    ds = secant(phi0,phi1,s[0],s[1])
    y = y[:,0] #taking out only the positive values
    s[0]  = s[1]
    s[1] +=  ds
    phi0 = phi1
    print('n = {}  s1 = {} and ds = {}'.format(i,s[1],ds))
    plt.plot(x, y, linestyles[i + 1])
    legendList.append('it = {0}'.format(i + 1))
    
    if (abs(ds)<=eps):
        print('Solution converged for eps = {} and s1 ={} and ds = {}. \n'.format(eps,s[1],ds))
        break



plt.xlabel('x')
plt.ylabel('y')
plt.legend(legendList, frameon=False, loc='best')
#plt.savefig('fig/Problem1_it.png', transparent=True)

plt.show()

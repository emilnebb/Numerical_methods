import numpy as np
import matplotlib.pyplot as plt
import scipy as sip
import sympy as sp
import scipy.special as sss

def eulers(x0,x1, h, y0, f): #for system of equations
    
    x= np.arange(x0,x1+h,h)
    y = np.zeros(np.size(x),float)

    for i in range(np.size(x)):
        y[i] = y0[0]        #the first element in y0 is theta, and the value we want
        y0 = y0 + h*f(y0)   #the Euler's step of computing next value of y
        x0 = x0 + h
    
    return x,y

def heuns(x0,x1, h, y0, f):

    x= np.arange(x0,x1+h,h)
    y = np.zeros(np.size(x),float)
    dy = np.zeros(np.size(x),float)

    for i in range(np.size(x)):
        y[i] = y0[0]        #the first element in y0 is theta, and the value we want
        dy[i] = y0[1]       #updating y-prime as well
        yp = y0 + h*f(y0)   #predictor-step
        y0 = y0 + (h/2)*(f(y0) + f(yp))   #corrector step
        x0 = x0 + h

    
    return x,y,dy


def f(x):             
    y = np.array([x[1], -(my/m)*x[1]-(g/l)*np.sin(x[0])])
    return y

def peaks(t, derivatives, y):
    
    t0 = 0
    increment = 0

    for i in range(1,np.size(derivatives)):
        if (derivatives[i-1]>0) and (derivatives[i]<0):
            increment = increment + 1

    peaks = np.zeros(increment+1, float)
    times = np.zeros(increment+1, float)
    times[0] = t0
    peaks[0] = y[0]
    inc = 1

    for i in range(1,np.size(derivatives)):
        if (derivatives[i-1]>0) and (derivatives[i]<0):
            T = t[i]-t0
            t0 = t[i]
            times[inc]= t0
            peaks[inc] = y[i]
            inc = inc + 1
            print (T)
    return times, peaks

#Defining constants and initial conditions:

theta0 = 85*np.pi/180
theta_prime = 0
g = 9.81
my = 1
m = 1
l = 1
dt = 0.01

y0 = np.array([theta0, theta_prime])
t0 = 0
t1 = 10

t, theta = eulers(t0, t1, dt, y0, f)
t_heuns, theta_heuns, deriverte = heuns(t0, t1, dt, y0, f)
tider, topper = peaks(t,deriverte, theta_heuns)

plt.plot(t,theta, label = "Eulers")
plt.plot(t_heuns, theta_heuns, label = "Heuns")
plt.plot(tider, topper, 'o', label = "topper")
plt.legend()
plt.grid()
plt.show()






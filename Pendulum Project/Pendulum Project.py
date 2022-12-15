import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from math import *
from sympy import *
from scipy.integrate import odeint

x,y,r,t,m,v,i,g= symbols('x,y,r,t,m,v,i,g')
theta = Function('theta')(t)

x = r * sin(theta)
y = (-1*r) * cos(theta)
theta_dot = diff(theta, t)

vx = diff(x,t)
vy = diff(y,t)

vt = vx**2 + vy**2
T_rec = 0.5* m * v.subs(v, vt)
T_rot = 0.5* i* diff(theta,t)**2

T = T_rec + T_rot
U = m*g*r*(1 - cos(theta))
L = T - U

Q = diff(diff(L, theta_dot),t) - diff(L, theta)

def ode(theta, t):
    a = (-m*g*r*(np.sin(theta)))/(i+(m*(r**2)))
    return a

m = 1
r  = 0.5
i = 0.025
theta0 = np.pi/2
t = np.linspace(0, 10, 100)
g = 9.81

theta = odeint(ode, theta0, t)
print(theta)

plt.plot(t, theta[:,0])
plt.show()
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from math import *
from sympy import *
from scipy.integrate import odeint

x,y,r,t,m,v,i,g= symbols('x,y,r,t,m,v,i,g')#type:ignore
theta = Function('theta')(t)
y = (-1*r) * cos(theta)
theta_dot = diff(theta, t)

vx = diff(x,t)
vy = diff(y,t)

vt = vx**2 + vy**2
T_rec = 0.5* m* v.subs(v, vt)
T_rot = 0.5* i* diff(theta,t)**2

T = T_rec + T_rot
U = m*g*r*(1 - cos(theta))# type: ignore
L = T - U

Q = diff(diff(L, theta_dot),t) - diff(L, theta)

def ode_func(theta,t,m,g,r,i):
    theta1=theta[0]
    theta2=theta[1]
    #first ode
    dtheta1_dt=theta2
    #second ode
    dtheta2_dt = (-m*g*r*(np.sin(theta1)))/(i+(m*(r**2)))
    dtheta_dt = [dtheta1_dt, dtheta2_dt]
    return dtheta_dt
g = 9.81 # acceleration due to the gravity
i = 0.025
r = 0.5 # radius of pendulum
m = 1 # mass 

# initial conditions
theta_0 = [(np.pi/2),0, (np.pi/2), (np.pi/2), (np.pi/4), 0, (np.pi/4)]
# time plot
t = np.linspace(0, 10, 1000)
#T = len(t)
# solving the ode
for a in range(len(theta_0)):
    theta = odeint(ode_func,theta_0[a:a+2],t,args=(m,g,r,i,a))

    plt.figure(figsize=(7.5, 10.5))
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'normal'
    plt.minorticks_on()
    plt.grid(visible=True, which='major', linestyle='-')
    plt.grid(visible=True, which='minor', linestyle='--')
    plt.xlim(0,10)

    plt.plot(t,theta[:,0],'--', color='k', label=r'$\dot{\mathrm{\theta}}$')
    plt.plot(t,theta[:,1], color='k', label=r'$\mathrm{\ddot{\theta}}$')
    plt.xlabel('t/s')
    plt.ylabel(r'$\Delta \theta$/rads')
    plt.title(r'A graph of the change in $\mathrm{\theta}$ in time')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Plot 1.{a+1}.png', dpi=800)
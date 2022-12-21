import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from math import *
from sympy import *
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq
import matplotlib.animation as ani

x,y,r,t,m,v,i,g = symbols('x,y,r,t,m,v,i,g')#type:ignore
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
    #theta
    theta1=theta[0]
    #theat_dot
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
l = 1 # length of pendulum  

def animate_arm(state,t):
    l = 1
    plt.plot(0,0,l)
    x = l * np.sin(state[0])
    y = -l * np.cos(state[0])
    p, = plt.plot(x, y, 'k-')
    tt=plt.title("{:.2} sec".format(0.00))
    plt.xlim([-0.25, 0.25])
    plt.ylim([-l, .10])
    for i in range(1, len(state)-10,3):
        p.set_xdata((0,l*np.sin(state[i])))
        p.set_ydata((0,-l*np.cos(state[i])))
        tt.set_text("{:.2} sec".format(i*0.01))
        plt.draw()
        plt.pause(0.1)


# initial conditions
theta_0 = [(np.pi/2),0, (np.pi/2), (np.pi/2), (np.pi/4), 0, (np.pi/4)]
# time plot
t = np.linspace(0, 10, 1000)

# solving the ode
for a in range(len(theta_0)-1):
    theta = odeint(ode_func,theta_0[a:a+2],t,args=(m,g,r,i))

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
    #plt.savefig(f'Plot 1.{a+1}.png', dpi=800)
    plt.close()

    # Fourier transformation
    # Sampling space = Frequency * total time
    N = 100 * 10
    # theta
    x = theta[:,0]
    # theta_dot
    y = theta[:,1]
    # Periodic Time (maybe ?)
    T = 1/N
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]

    # Plotting conditions
    plt.figure(figsize=(7.5, 10.5))
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'normal'
    plt.minorticks_on()
    plt.grid(visible=True, which='major', linestyle='-')
    plt.grid(visible=True, which='minor', linestyle='--')
    # Plotting the data
    plt.plot(xf, 2/N * np.abs(yf[0:(N//2)]))#type:ignore
    plt.xlabel('Frequency')
    plt.ylabel(r'$\Delta \theta$/rads')
    plt.title('Fourier transformation of the Single Pendulum')
    plt.tight_layout()
    # Saving the figure
    #plt.savefig(f'Plot 2.{a+1}.png', dpi=800)
    plt.close()

plt.figure()
animate_arm(theta[:,0],t)
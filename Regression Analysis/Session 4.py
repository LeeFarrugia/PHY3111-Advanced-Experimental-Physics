#Task 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns

data = pd.read_csv('Q4__Galilean_Moon_Astrometric_Data.csv')

io_position = data['Io_Offset (Jup Diameters)']
europa_position = data['Europa_Offset (Jup Diameters)']
ganymede_position = data['Ganymede_Offset (Jup Diameters)']
callisto_position = data['Callisto_Offset (Jup Diameters)']

io_hjd = data['Io_Julian_Date (HJD)']
europa_hjd = data['Europa_Julian_Date (HJD)']
ganymede_hjd = data['Ganymede_Julian_Date (HJD)']
callisto_hjd = data['Callisto_Julian_Date (HJD)']

io_lin = np.linspace(io_hjd.min(), io_hjd.max(), 1000)
europa_lin = np.linspace(europa_hjd.min(), europa_hjd.max(), 1000)
ganymede_lin = np.linspace(ganymede_hjd.min(), ganymede_hjd.max(), 1000)
callisto_lin = np.linspace(callisto_hjd.min(), callisto_hjd.max(), 1000)

def wave_func(t, A, T):
    return A*np.sin(((2*np.pi)/T)*t)

popt_io, pcov_io = curve_fit(wave_func, io_hjd, io_position, p0=(max(io_position), 1.75))
fitted_io = wave_func(io_lin, popt_io[0], popt_io[1])

popt_europa, pcov_europa = curve_fit(wave_func, europa_hjd, europa_position, p0=(max(europa_position),3.56))
fitted_europa = wave_func(europa_lin, popt_europa[0], popt_europa[1])

popt_ganymede, pcov_ganymede = curve_fit(wave_func, ganymede_hjd, ganymede_position, p0=(max(ganymede_position), 7.15))
fitted_ganymede = wave_func(ganymede_lin, popt_ganymede[0], popt_ganymede[1])

popt_callisto, pcov_callisto = curve_fit(wave_func, callisto_hjd, callisto_position, p0=(max(callisto_position), 16.5))
fitted_callisto = wave_func(callisto_lin, popt_callisto[0], popt_callisto[1])

f, (a0, a1, a2, a3) = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(7.3, 10.7))

a0.minorticks_on()
a0.grid(visible=True, which='major', linestyle='-')
a0.grid(visible=True, which='minor', linestyle='--')
a0.set_xlabel('Julian Date')
a0.set_ylabel('Offset in Jupiter Diameter')
a0.scatter(io_hjd, io_position, color='k', label='Io')
a0.plot(io_lin, fitted_io, '--', color='k', label='Io Fit')
a0.legend()

a1.minorticks_on()
a1.grid(visible=True, which='major', linestyle='-')
a1.grid(visible=True, which='minor', linestyle='--')
a1.set_xlabel('Julian Date')
a1.set_ylabel('Offset in Jupiter Diameter')
a1.scatter(europa_hjd, europa_position, marker='s', color='k', label='Europa') # type:ignore
a1.plot(europa_lin, fitted_europa, '--', color='k', label='Europa Fit')
a1.legend()

a2.minorticks_on()
a2.grid(visible=True, which='major', linestyle='-')
a2.grid(visible=True, which='minor', linestyle='--')
a2.set_xlabel('Julian Date')
a2.set_ylabel('Offset in Jupiter Diameter')
a2.scatter(ganymede_hjd, ganymede_position, color='k', label='Ganymede')
a2.plot(ganymede_lin, fitted_ganymede, '--', color='k', label='Ganymede Fit')
a2.legend()

a3.minorticks_on()
a3.grid(visible=True, which='major', linestyle='-')
a3.grid(visible=True, which='minor', linestyle='--')
a3.set_xlabel('Julian Date')
a3.set_ylabel('Offset in Jupiter Diameter')
a3.scatter(callisto_hjd, callisto_position, marker='^', color='k', label='Callisto')  # type:ignore
a3.plot(callisto_lin, fitted_callisto, '--', color='k', label='Callisto Fit')
a3.legend()

f.tight_layout()
#f.savefig('4Plot1.png', dpi=800)
#plt.show()
plt.close()

io_rad = abs(popt_io[0])*138920000
europa_rad = abs(popt_europa[0])*138920000
ganymede_rad = abs(popt_ganymede[0])*138920000
callisto_rad = abs(popt_callisto[0])*138920000

print(f'Io semi-major axis is: {io_rad:.2}m, Europa semi-major axis is: {europa_rad:.2}m, Ganymede semi-major axis is: {ganymede_rad:.2}m, Callisto semi-major axis is: {callisto_rad:.2}m')

# 0.16 is a factor to convert from julian date to normal days
io_period = abs(popt_io[1])*86400
europa_period = abs(popt_europa[1])*86400
ganymede_period = abs(popt_ganymede[1])*86400
callisto_period = abs(popt_callisto[1])*86400

print(f'Io period is: {io_period:.2f}s, Europa period is: {europa_period:.2f}s, Ganymede period is: {ganymede_period:.2f}s, Callisto period is: {callisto_period:.2f}s')

radius = np.array([io_rad, europa_rad, ganymede_rad, callisto_rad])
period = np.array([io_period, europa_period, ganymede_period, callisto_period])

Y = radius**3
X = period**2

coeffs, cov = np.polyfit(X, Y, 1, cov=True)
poly_function = np.poly1d(coeffs)
fit_line = poly_function(X)

plt.figure(figsize=(7.5,10.5))
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.scatter(X, Y, color='k')
plt.plot(X, fit_line, '-', color='k')
#plt.savefig('4Plot2.png', dpi=800)
#plt.show()

grad = coeffs[0]
G = 6.6743e-11
jupiter_mass = (4*(np.pi**2)*grad)/G
print(f'The mass of jupiter is: {jupiter_mass:.2E}kg')

r2_io = io_rad**2
r2_europa = europa_rad**2
r2_ganymede = ganymede_rad**2
r2_callisto = callisto_rad**2

F_io = 6.35e22
F_europa = 1.4e22
F_ganymede = 1.63e22
F_callisto = 3.87e21

m_io = (F_io*(r2_io))/(G*jupiter_mass)
m_europa = (F_europa*(r2_europa))/(G*jupiter_mass)
m_ganymede = (F_ganymede*(r2_ganymede))/(G*jupiter_mass)
m_callisto = (F_callisto*(r2_callisto))/(G*jupiter_mass)

print(f'The mass of Io is: {m_io:.2E}kg, the mass of Europa is: {m_europa:.2E}kg, the mass of Ganymede is: {m_ganymede:.2E}kg, the mass of callisto is: {m_callisto:.2E}kg')

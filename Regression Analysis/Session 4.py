#Task 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns

# importing the data to be analysed
data = pd.read_csv('Q4__Galilean_Moon_Astrometric_Data.csv')

io = data['Io_Offset (Jup Diameters)']
europa = data['Europa_Offset (Jup Diameters)']
ganymede = data['Ganymede_Offset (Jup Diameters)']
callisto = data['Callisto_Offset (Jup Diameters)']

hjd_io = data['Io_Julian_Date (HJD)']
hjd_europa = data['Europa_Julian_Date (HJD)']
hjd_ganymede = data['Ganymede_Julian_Date (HJD)']
hjd_callisto = data['Callisto_Julian_Date (HJD)']

hjd_lin_io = np.linspace(hjd_io.min()-1, hjd_io.max()+1, 1000)
hjd_lin_europa = np.linspace(hjd_europa.min()-1, hjd_europa.max()+1, 1000)
hjd_lin_ganymede = np.linspace(hjd_ganymede.min()-1, hjd_ganymede.max()+1, 1000)
hjd_lin_callisto = np.linspace(hjd_callisto.min()-1, hjd_callisto.max()+1, 1000)


def wave_function(t, A, w):
    return A * np.sin(w*t)

popt_io, pcov_io = curve_fit(wave_function, hjd_io, io, p0=(max(io), 3.59))
fitted_line_io = wave_function(hjd_lin_io, popt_io[0], popt_io[1])

popt_europa, pcov_europa = curve_fit(wave_function, hjd_europa, europa, p0=(max(europa), 4))
fitted_line_europa = wave_function(hjd_lin_europa, popt_europa[0], popt_europa[1])

popt_ganymede, pcov_ganymede = curve_fit(wave_function, hjd_io, io, p0=(max(ganymede), 8))
fitted_line_ganymede = wave_function(hjd_lin_ganymede, popt_ganymede[0], popt_ganymede[1])

popt_callisto, pcov_callisto = curve_fit(wave_function, hjd_callisto, callisto, p0=(max(callisto), 16))
fitted_line_callisto = wave_function(hjd_lin_callisto, popt_callisto[0], popt_callisto[1])

f, (a0, a1, a2, a3) = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(7.3, 10.7))

a0.minorticks_on()
a0.grid(visible=True, which='major', linestyle='-')
a0.grid(visible=True, which='minor', linestyle='--')
a0.set_xlabel('Julian Date')
a0.set_ylabel('Offset in Jupiter Diameter')
a0.scatter(hjd_io, io, color='k', label='Io')
a0.plot(hjd_lin_io, fitted_line_io, '--', color='k', label='Io Fit')
a0.legend()

a1.minorticks_on()
a1.grid(visible=True, which='major', linestyle='-')
a1.grid(visible=True, which='minor', linestyle='--')
a1.set_xlabel('Julian Date')
a1.set_ylabel('Offset in Jupiter Diameter')
a1.scatter(hjd_europa, europa, marker='s', color='k', label='Europa') # type:ignore
a1.plot(hjd_lin_europa, fitted_line_europa, '--', color='k', label='Europa Fit')
a1.legend()

a2.minorticks_on()
a2.grid(visible=True, which='major', linestyle='-')
a2.grid(visible=True, which='minor', linestyle='--')
a2.set_xlabel('Julian Date')
a2.set_ylabel('Offset in Jupiter Diameter')
a2.scatter(hjd_ganymede, ganymede, color='k', label='Ganymede')
a2.plot(hjd_lin_ganymede, fitted_line_ganymede, '--', color='k', label='Ganymede Fit')
a2.legend()

a3.minorticks_on()
a3.grid(visible=True, which='major', linestyle='-')
a3.grid(visible=True, which='minor', linestyle='--')
a3.set_xlabel('Julian Date')
a3.set_ylabel('Offset in Jupiter Diameter')
a3.scatter(hjd_callisto, callisto, marker='^', color='k', label='Callisto')  # type:ignore
a3.plot(hjd_lin_callisto, fitted_line_callisto, '--', color='k', label='Callisto Fit')
a3.legend()

f.tight_layout()
#f.savefig('4Plot1.png', dpi=800)
#plt.show()
plt.close()

plt.scatter(hjd_europa, europa)
plt.plot(hjd_europa, europa)
plt.show()

quit()

io_rad = abs(popt_io[0])*138920000
europa_rad = abs(popt_europa[0])*138920000
ganymede_rad = abs(popt_ganymede[0])*138920000
callisto_rad = abs(popt_callisto[0])*138920000

print(f'Io semi-major axis is: {io_rad:.2}m, Europa semi-major axis is: {europa_rad:.2}m, Ganymede semi-major axis is: {ganymede_rad:.2}m, Callisto semi-major axis is: {callisto_rad:.2}m')

# 0.16 is a factor to convert from julian date to normal days
io_period = abs(popt_io[1])*0.16*86400
europa_period = abs(popt_europa[1])*0.16*86400
ganymede_period = abs(popt_ganymede[1])*0.16*86400
callisto_period = abs(popt_callisto[1])*0.16*86400

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
# plt.savefig('4Plot2.png', dpi=800)
#plt.show()

grad = coeffs[0]
G = 6.6743e-11
jupiter_mass = (4*(np.pi**2)*grad)/G
print(f'The mass of jupiter is: {jupiter_mass:.2E}kg')

a = 24.79
r2_io = io_rad**2
r2_europa = europa_rad**2
r2_ganymede = ganymede_rad**2
r2_callisto = callisto_rad**2

Fg = jupiter_mass * a

m_io = (Fg*(r2_io))/(G*jupiter_mass)
m_europa = (Fg*(r2_europa))/(G*jupiter_mass)
m_ganymede = (Fg*(r2_ganymede))/(G*jupiter_mass)
m_callisto = (Fg*(r2_callisto))/(G*jupiter_mass)

# m_io = (4*(np.pi**2)*(io_rad**3)*(io_period**2))/(G)
# m_europa = (4*(np.pi**2)*(europa_rad**3)*(europa_period**2))/(G)
# m_ganymede = (4*(np.pi**2)*(ganymede_rad**3)*(ganymede_period**2))/(G)
# m_callisto = (4*(np.pi**2)*(callisto_rad**3)*(callisto_period**2))/(G)

print(f'The mass of Io is: {m_io:.2E}kg, the mass of Europa is: {m_europa:.2E}kg, the mass of Ganymede is: {m_ganymede:.2E}kg, the mass of callisto is: {m_callisto:.2E}kg')

#Task 4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# importing the data 
data = pd.read_csv('Q4__Galilean_Moon_Astrometric_Data.csv')

# importing the offset data for each moon
io_position = data['Io_Offset (Jup Diameters)']
europa_position = data['Europa_Offset (Jup Diameters)']
ganymede_position = data['Ganymede_Offset (Jup Diameters)']
callisto_position = data['Callisto_Offset (Jup Diameters)']

# importing the hjd data for each moon
io_hjd = data['Io_Julian_Date (HJD)']
europa_hjd = data['Europa_Julian_Date (HJD)']
ganymede_hjd = data['Ganymede_Julian_Date (HJD)']
callisto_hjd = data['Callisto_Julian_Date (HJD)']

# creating linspaces for each moon to obtain a smooth curve
io_lin = np.linspace(io_hjd.min(), io_hjd.max(), 1000)
europa_lin = np.linspace(europa_hjd.min(), europa_hjd.max(), 1000)
ganymede_lin = np.linspace(ganymede_hjd.min(), ganymede_hjd.max(), 1000)
callisto_lin = np.linspace(callisto_hjd.min(), callisto_hjd.max(), 1000)

# defining the wave function to plot the data
def wave_func(t, A, T):
    return A*np.sin(((2*np.pi)/T)*t)

# determining the curve fit to the io data and obtaining the line data
popt_io, pcov_io = curve_fit(wave_func, io_hjd, io_position, p0=(max(io_position), 1.75))
fitted_io = wave_func(io_lin, popt_io[0], popt_io[1])

# determining the curve fit to the europa data and obtaining the line data
popt_europa, pcov_europa = curve_fit(wave_func, europa_hjd, europa_position, p0=(max(europa_position),3.56))
fitted_europa = wave_func(europa_lin, popt_europa[0], popt_europa[1])

# determining the curve fit to the ganymede data and obtaining the line data
popt_ganymede, pcov_ganymede = curve_fit(wave_func, ganymede_hjd, ganymede_position, p0=(max(ganymede_position), 7.15))
fitted_ganymede = wave_func(ganymede_lin, popt_ganymede[0], popt_ganymede[1])

# determining the curve fit to the callisto data and obtaining the line data
popt_callisto, pcov_callisto = curve_fit(wave_func, callisto_hjd, callisto_position, p0=(max(callisto_position), 16.5))
fitted_callisto = wave_func(callisto_lin, popt_callisto[0], popt_callisto[1])

# defining the subplots to be plotted with their respective data, size and variables
f, (a0, a1, a2, a3) = plt.subplots(4, 1, sharex=False, sharey=False, figsize=(7.3, 10.7))

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

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
# saving the figure
f.savefig('4Plot1.png', dpi=800)
plt.show()
# clearing the plotted figure
plt.close()

# calculating the semi-major axis of each moon in meters
io_rad = abs(popt_io[0])*138920000
europa_rad = abs(popt_europa[0])*138920000
ganymede_rad = abs(popt_ganymede[0])*138920000
callisto_rad = abs(popt_callisto[0])*138920000

# displaying the semi-major axis of each moon
print(f'Io semi-major axis is: {io_rad:.2}m, Europa semi-major axis is: {europa_rad:.2}m, Ganymede semi-major axis is: {ganymede_rad:.2}m, Callisto semi-major axis is: {callisto_rad:.2}m')

# calculating the periodic time of each moon in seconds
io_period = abs(popt_io[1])*86400
europa_period = abs(popt_europa[1])*86400
ganymede_period = abs(popt_ganymede[1])*86400
callisto_period = abs(popt_callisto[1])*86400

# displaying the periodic time for each moon
print(f'Io period is: {io_period:.2f}s, Europa period is: {europa_period:.2f}s, Ganymede period is: {ganymede_period:.2f}s, Callisto period is: {callisto_period:.2f}s')

# defining arrays for radii and periods
radius = np.array([io_rad, europa_rad, ganymede_rad, callisto_rad])
period = np.array([io_period, europa_period, ganymede_period, callisto_period])

# finding r^3 and T^2
Y = radius**3
X = period**2

# determining the line of best fit for the given data
coeffs, cov = np.polyfit(X, Y, 1, cov=True)
poly_function = np.poly1d(coeffs)
fit_line = poly_function(X)

# plotting the straight line graph
plt.figure(figsize=(7.5,10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.scatter(X, Y, color='k')
plt.plot(X, fit_line, '-', color='k')
plt.ylabel(r'r$^3$/m$^3$')
plt.xlabel(r'T$^2$/s$^2$')
plt.title(r'A Graph of r$^3$ vs T$^2$')
plt.savefig('4Plot2.png', dpi=800)
plt.show()

# determining the gradient of the straight line and the gradient error
grad = coeffs[0]
grad_err = np.sqrt(cov[0][0])
# defining the gravitational constant and real jupiter mass
G = 6.6743e-11
jupiter_real_mass =1.898e27
# finding the mass of jupiter
jupiter_mass = (4*(np.pi**2)*grad)/G
# finding the error of the mass of jupiter calculation
delta_jupiter_mass = np.sqrt((((4*(np.pi**2))/G)*(grad_err))**2)
jupiter_precision = abs(((jupiter_mass/jupiter_real_mass)-1)*100)
# displaying the results
print(f'The mass of jupiter is: {jupiter_mass:.2E}kg ± {delta_jupiter_mass:.2E} an precision of {jupiter_precision:.2f}%')

# the semi-major axis squared
r2_io = io_rad**2
r2_europa = europa_rad**2
r2_ganymede = ganymede_rad**2
r2_callisto = callisto_rad**2

# real masses of jupiter's moons
real_io_mass = 8.932e22
real_europa_mass = 4.8e22
real_ganymede_mass = 1.482e23
real_callisto_mass = 1.076e23

# gravitational force of jupiter on the respective moon
F_io = 6.35e22
F_europa = 1.4e22
F_ganymede = 1.63e22
F_callisto = 3.87e21

# finding the mass of each moon
m_io = (F_io*(r2_io))/(G*jupiter_mass)
m_europa = (F_europa*(r2_europa))/(G*jupiter_mass)
m_ganymede = (F_ganymede*(r2_ganymede))/(G*jupiter_mass)
m_callisto = (F_callisto*(r2_callisto))/(G*jupiter_mass)

# calculating the precision for the masses of the moons
io_precision = abs(((m_io/real_io_mass)-1)*100)
europa_precision = abs(((m_europa/real_europa_mass)-1)*100)
ganymede_precision = abs(((m_ganymede/real_ganymede_mass)-1)*100)
callisto_precision = abs(((m_callisto/real_callisto_mass)-1)*100)

# displaying the values obtained
print(f'The mass of Io is: {m_io:.2E}kg with a precision of {io_precision:.2f}%. The mass of Europa is: {m_europa:.2E}kg with a precision of {europa_precision:.2f}%. The mass of Ganymede is: {m_ganymede:.2E}kg with a precision of {ganymede_precision:.2f}%. The mass of callisto is: {m_callisto:.2E}kg with a precision of {callisto_precision:.2f}%.')

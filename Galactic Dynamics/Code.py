import numpy as np
import matplotlib.pyplot as plt
import pynbody
import pynbody.plot.sph as sph
from functools import reduce

s1 = pynbody.load('run708main.01000')

print(s1.families()) # Exploring groups in simulation
print(s1.properties) # Showing the properties of the simulation
print(f's1.s: {s1.s.loadable_keys()}') # Showing keys for stars
print(f's1.d: {s1.d.loadable_keys()}') # Showing keys for dark matter
print(f's1.g: {s1.g.loadable_keys()}') # Showing keys for gas

s2 = pynbody.load('run708mainDiff.01000')

# print(s2.families()) # Exploring groups in simulation
# print(s2.properties) # Showing the properties of the simulation
# print(f's2.s: {s2.s.loadable_keys()}') # Showing keys for stars
# print(f's2.d: {s2.d.loadable_keys()}') # Showing keys for dark matter
# print(f's2.g: {s2.g.loadable_keys()}') # Showing keys for gas

# #Question 1
s1.physical_units() # Changing units to correct units
pynbody.analysis.angmom.faceon(s1)


s2.physical_units() # Changing units to correct units
pynbody.analysis.angmom.faceon(s2)

# #Question 2
# mass_s1_s = s1.s['mass']
# mass_s1_d = s1.d['mass']
# mass_s1_g = s1.g['mass']

# mass_s2_s = s2.s['mass']
# mass_s2_d = s2.d['mass']
# mass_s2_g = s2.g['mass']

# Sum = 0
# array = []

# def mass_sum(array):
#   Sum = reduce(lambda a, b: a+b, array)
#   return(Sum)

# total_mass_s1 = mass_sum(mass_s1_s) + mass_sum(mass_s1_d) + mass_sum(mass_s1_g)

# Sum = 0
# array = []

# total_mass_s2 = mass_sum(mass_s2_s) + mass_sum(mass_s2_d) + mass_sum(mass_s2_g)

# print(f'Total mass s1: {total_mass_s1}')
# print(f'Total mass s2: {total_mass_s2}')

# #Question 3
# plt.figure(figsize=(7.5,10.5))
# pynbody.plot.image(s1.s, threaded=False)
# # plt.savefig(f'Plots/Figure 1.png', dpi=800)
# # plt.clf

# plt.figure(figsize=(7.5,10.5))
# pynbody.plot.stars.render(s1.s)
# # plt.savefig(f'Plots/Figure 2.png', dpi=800)
# # plt.clf

# plt.figure(figsize=(7.5,10.5))
# pynbody.plot.image(s2.s, threaded=False)
# # plt.savefig(f'Plots/Figure 3.png', dpi=800)
# # plt.clf

# plt.figure(figsize=(7.5,10.5))
# pynbody.plot.stars.render(s2.s)
# # plt.savefig(f'Plots/Figure 4.png', dpi=800)
# # plt.clf


# #Question 4
# plt.figure(figsize=(7.5,10.5))
# pynbody.plot.image(s1.g, threaded=False)
# # plt.savefig(f'Plots/Figure 5.png', dpi=800)
# # plt.clf

# plt.figure(figsize=(7.5,10.5))
# pynbody.plot.image(s2.g, threaded=False)
# # plt.savefig(f'Plots/Figure 6.png', dpi=800)
# # plt.clf

# #Question 6
plt.figure(figsize=(7.5,10.5))

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')

p1 = pynbody.analysis.profile.Profile(s1.s, max=30, nbins=200, ndim=3)
plt.plot(p1['rbins'], p1['rho'], color='k')
plt.ylabel(r'$\log{\rho}$')
plt.semilogy()
plt.xlabel('Stellar Radii')
plt.title('Stellar radial density profile of system 1')
# plt.savefig(f'Plots/Figure 7.png', dpi=800)
# plt.clf

plt.figure(figsize=(7.5,10.5))

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')

p2 = pynbody.analysis.profile.Profile(s2.s, max=40, nbins=250, ndim=3)
plt.plot(p2['rbins'], p2['rho'], color='k')
plt.ylabel(r'$\log{\rho}$')
plt.semilogy()
plt.xlabel('Stellar Radii')
plt.title('Stellar radial density profile of system 2')
plt.show()
# plt.savefig(f'Plots/Figure 8.png', dpi=800)
# plt.clf

#Task2

#Question 1
pynbody.analysis.angmom.sideon(s1)
pynbody.plot.image(s1.s, threaded=False)
plt.savefig(f'Plots/Figure 9.png', dpi=800)
plt.clf

#Question 2
radius = 4 # Defining the radius to be considered 
centre = (0,0,0) # Defining the center of the galaxy

sphere1 = s1.s[pynbody.filt.Sphere(radius, centre)] # Filtering the stars according to radii
pynbody.analysis.angmom.sideon(sphere1) # Aligning the data for a side view

sph.image(s1.s, width='8 kpc') # Generating a density heat map of the filtered stars
plt.savefig(f'Plots/Figure 10.png', dpi=800)
plt.clf
pynbody.plot.stars.render(sphere1, width='8 kpc') # Generating an image of the filtered stars
plt.savefig(f'Plots/Figure 11.png', dpi=800)
plt.clf

sphere2 = s2.s[pynbody.filt.Sphere(radius, centre)] # Filtering the stars according to radii
pynbody.analysis.angmom.sideon(sphere2)

sph.image(s2.s, width='8 kpc') # Generating a density heat map of the filtered stars
plt.savefig(f'Plots/Figure 12.png', dpi=800)
plt.clf
pynbody.plot.stars.render(sphere2, width='8 kpc') # Generating an image of the filtered stars
plt.savefig(f'Plots/Figure 13.png', dpi=800)
plt.clf

#Question 3
pynbody.analysis.angmom.sideon(s1) # Viewing the galaxy from the side
sph.image(s1.s, width='1.5 kpc') # Generating density heat map
plt.savefig(f'Plots/Figure 14.png', dpi=800)
plt.clf
pynbody.plot.stars.render(s1.s, width='1.5 kpc') # Generating an image, change the width in step of 0.25 from 0 to 1.5
plt.savefig(f'Plots/Figure 15.png', dpi=800)
plt.clf

pynbody.analysis.angmom.sideon(s2) # Viewing the galaxy from the side
sph.image(s2.s, width='1.5 kpc') # Generating density heat map
plt.savefig(f'Plots/Figure 16.png', dpi=800)
plt.clf
pynbody.plot.stars.render(s2.s, width='1.5 kpc') # Generating an image, change the width in step of 0.25 from 0 to 1.5
plt.savefig(f'Plots/Figure 17.png', dpi=800)
plt.clf

#Question 4
rho_s1 = pynbody.analysis.profile.Profile(s1.s, rmin=0, rmax=1.5) # Filtering stars according to their radii, change rmax in steps of 0.25 from 0 to 1.5

plt.figure(figsize=(7.5,10.5))

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')

plt.plot(rho_s1['rbins'], np.log(rho_s1['rho']), color='k') # Plotting the radii density profile

plt.ylabel(r'$\log{\rho}$')
plt.xlabel('Stellar Radii')
plt.title('Stellar radial density profile of system 1')
plt.savefig(f'Plots/Figure 18.png', dpi=800)
plt.clf

rho_s2 = pynbody.analysis.profile.Profile(s2.s, rmin=0, rmax=1.5) # Filtering stars according to their radii, change rmax in steps of 0.25 from 0 to 1.5

plt.figure(figsize=(7.5,10.5))

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')

plt.plot(rho_s2['rbins'], np.log(rho_s2['rho']), color='k') # Plotting the radii density profile

plt.ylabel(r'$\log{\rho}$')
plt.xlabel('Stellar Radii')
plt.title('Stellar radial density profile of system 2')
plt.savefig(f'Plots/Figure 19.png', dpi=800)
plt.clf

#Question 5
radius_filter = pynbody.filt.BandPass('pos', '0.75 kpc', '1 kpc')
s1_filtered = s1.s[radius_filter]
s1_p = pynbody.analysis.profile.Profile(s1_filtered, nbins=200)

plt.figure(figsize=(7.5,10.5))

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')

plt.clf; plt.plot(s1_p['rbins'], np.log(s1_p['rho']), color ='k')

plt.ylabel(r'$\log{\rho}$')
plt.xlabel('Stellar Ages')
plt.title('Stellar radial density profile of system 1')
plt.savefig(f'Plots/Figure 20.png', dpi=800)
plt.clf

s2_filtered = s2.s[radius_filter]
s2_p = pynbody.analysis.profile.Profile(s2_filtered, nbins=200)

plt.figure(figsize=(7.5,10.5))

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')

plt.clf; plt.plot(s2_p['rbins'], np.log(s2_p['rho']), color ='k')

plt.ylabel(r'$\log{\rho}$')
plt.xlabel('Stellar Ages')
plt.title('Stellar radial density profile of system 1')
plt.savefig(f'Plots/Figure 21.png', dpi=800)
plt.clf

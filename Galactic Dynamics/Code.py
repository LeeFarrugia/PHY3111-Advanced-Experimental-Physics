import numpy as np
import matplotlib.pyplot as plt
import pynbody
import pynbody.plot.sph as sph
from functools import reduce

s1 = pynbody.load('run708main.01000')

print(s1.families()) # Exploring groups in simulation
print(s1.properties) # Showing the properties of the simulation
print(f's1.s: {s1.s.loadable_keys()}') # Showing keys for stars
print(f's1.d: {s1.d.loadable_keys()}') # Showing keys for distance modulus
print(f's1.g: {s1.g.loadable_keys()}') # Showing keys for gas

s2 = pynbody.load('run708mainDiff.01000')

print(s2.families()) # Exploring groups in simulation
print(s2.properties) # Showing the properties of the simulation
print(f's2.s: {s2.s.loadable_keys()}') # Showing keys for stars
print(f's2.d: {s2.d.loadable_keys()}') # Showing keys for distance modulus
print(f's2.g: {s2.g.loadable_keys()}') # Showing keys for gas

#Question 1
s1.physical_units() # Changing units to correct units
pynbody.analysis.angmom.faceon(s1)


s2.physical_units() # Changing units to correct units
pynbody.analysis.angmom.faceon(s2)

#Question 2
mass_s1_s = s1.s['mass']
mass_s1_d = s1.d['mass']
mass_s1_g = s1.g['mass']

mass_s2_s = s2.s['mass']
mass_s2_d = s2.d['mass']
mass_s2_g = s2.g['mass']

Sum = 0
array = []

def mass_sum(array):
  Sum = reduce(lambda a, b: a+b, array)
  return(Sum)

total_mass_s1 = mass_sum(mass_s1_s) + mass_sum(mass_s1_d) + mass_sum(mass_s1_g)

Sum = 0
array = []

total_mass_s2 = mass_sum(mass_s2_s) + mass_sum(mass_s2_d) + mass_sum(mass_s2_g)

print(f'Total mass s1: {total_mass_s1}')
print(f'Total mass s2: {total_mass_s2}')

#Question 3

pynbody.plot.image(s1.s, threaded=False)
plt.clf; plt.show()

pynbody.plot.stars.render(s1.s)
plt.clf; plt.show()

pynbody.plot.image(s2.s, threaded=False)
plt.clf; plt.show()

pynbody.plot.stars.render(s2.s)
plt.clf; plt.show()


#Question 4
pynbody.plot.image(s1.g, threaded=False)
plt.clf; plt.show()

pynbody.plot.image(s2.g, threaded=False)
plt.clf; plt.show()

#Question 6
##To add: Title, x-axis, y-axis, grid...

p1 = pynbody.analysis.profile.Profile(s1.s)
plt.clf; plt.plot(p1['rbins'], np.log(p1['rho']))
plt.show()

p2 = pynbody.analysis.profile.Profile(s2.s)
plt.clf; plt.plot(p2['rbins'], np.log(p2['rho']))
plt.show()

#Task2

#Question 1
pynbody.analysis.angmom.sideon(s1)
pynbody.plot.image(s1.s, threaded=False)
plt.clf; plt.show()

#Question 2
radius = 4 # Defining the radius to be considered 
centre = (0,0,0) # Defining the center of the galaxy

sphere1 = s1.s[pynbody.filt.Sphere(radius, centre)] # Filtering the stars according to radii
pynbody.analysis.angmom.sideon(sphere1) # Aligning the data for a side view

sph.image(s1.s, width='8 kpc') # Generating a density heat map of the filtered stars
plt.clf; plt.show()
pynbody.plot.stars.render(sphere1, width='8 kpc') # Generating an image of the filtered stars
plt.clf; plt.show()

sphere2 = s2.s[pynbody.filt.Sphere(radius, centre)] # Filtering the stars according to radii
pynbody.analysis.angmom.sideon(sphere2)

sph.image(s2.s, width='8 kpc') # Generating a density heat map of the filtered stars
plt.clf; plt.show()
pynbody.plot.stars.render(sphere2, width='8 kpc') # Generating an image of the filtered stars
plt.clf; plt.show()

#Question 3

pynbody.analysis.angmom.sideon(s1) # Viewing the galaxy from the side
sph.image(s1.s, width='1.5 kpc') # Generating density heat map
plt.clf; plt.show()
pynbody.plot.stars.render(s1.s, width='1.5 kpc') # Generating an image, change the width in step of 0.25 from 0 to 1.5
plt.clf; plt.show()

pynbody.analysis.angmom.sideon(s2) # Viewing the galaxy from the side
sph.image(s2.s, width='1.5 kpc') # Generating density heat map
plt.clf; plt.show()
pynbody.plot.stars.render(s2.s, width='1.5 kpc') # Generating an image, change the width in step of 0.25 from 0 to 1.5
plt.clf; plt.show()

#Question 4
rho_s1 = pynbody.analysis.profile.Profile(s1.s, rmin=0, rmax=1.5) # Filtering stars according to their radii, change rmax in steps of 0.25 from 0 to 1.5
plt.clf; plt.plot(rho_s1['rbins'], np.log(rho_s1['rho'])) # Plotting the radii density profile
plt.show() # Showing plot

rho_s2 = pynbody.analysis.profile.Profile(s2.s, rmin=0, rmax=1.5) # Filtering stars according to their radii, change rmax in steps of 0.25 from 0 to 1.5
plt.clf; plt.plot(rho_s2['rbins'], np.log(rho_s2['rho'])) # Plotting the radii density profile
plt.show() # Showing plot

##Question 5
radius_filter = pynbody.filt.BandPass('pos', '0.75 kpc', '1 kpc')
s1_filtered = s1.s[radius_filter]
s1_p = pynbody.analysis.profile.Profile(s1_filtered, nbins=200)
plt.clf; plt.plot(s1_p['rbins'], np.log(s1_p['rho']))
plt.show()
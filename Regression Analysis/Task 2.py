#Task 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from math import pi

# creating empty list
A_list=[]

# importing the data to be analysed
data = pd.read_csv('Q2a__HR_Diagram.csv')

# grouping the data given by star type
data.groupby(['Star type'])

# setting parameters for plotting
plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')

# plotting scatter plot for the data given
plt.scatter((data['Temperature/K']), data['Luminosity(L/Lo)'], color='k')
plt.ylabel('L/Lo')
plt.xlabel('T/K')
plt.title('A graph of Luminosity vs Temperature')
plt.savefig('2Plot1.png', dpi=800)
plt.close()

# setting parameters for plotting
plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')


# plotting log of the data
plt.scatter(np.log(data['Temperature/K']), np.log(data['Luminosity(L/Lo)']), color='k')
plt.ylabel(r'$\log{L/Lo}$')
plt.xlabel(r'$\log{T/K}$')
plt.title(r'A graph of $\log{Luminosity}$ vs $\log{Temperature}$')
plt.savefig('2Plot2.png', dpi=800)
plt.close()

# setting parameters for plotting
plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')

# dropping the unwanted values
value_3 = data.mask(data['Star type']!=3).dropna().reset_index()
plt.scatter(np.log(value_3['Temperature/K']), np.log(value_3['Luminosity(L/Lo)']), color='k')
plt.ylabel(r'$\log{L/Lo}$')
plt.xlabel(r'$\log{T/K}$')
plt.title(r'A graph of $\log{\mathrm{Luminosity}}$ vs $\log{\mathrm{Temperature}}$ for star type 3')
plt.savefig('2Plot3.png', dpi=800)
plt.close()

# listing a number of the tried degree values
degrees = np.array([2, 10, 20, 30, 25, 15, 16, 17])
# creating a loop to test each degree until the smallest rmse is obtained and plotting each test

for i in degrees:
    # obtaining the log of the wanted data
    y = np.log(value_3['Luminosity(L/Lo)'])
    x = np.log(value_3['Temperature/K'])

    # reshaping the array
    x_a = x.array.reshape(-1,1)  # type: ignore
    poly = pf(degree=i)
    poly_Lumen=poly.fit_transform(x_a)

    model = LinearRegression()
    model.fit(poly_Lumen, y)
    y_pred = model.predict(poly_Lumen)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print(f'The root mean square is: {rmse}, with the degree of freedom is: {i}')

    plt.figure(figsize=(7.5, 10.5))
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'normal'
    plt.minorticks_on()
    plt.grid(visible=True, which='major', linestyle='-')
    plt.grid(visible=True, which='minor', linestyle='--')

    plt.scatter(x, y, color='k')
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x, y_pred), key=sort_axis)
    x, y_pred = zip(*sorted_zip)
    plt.plot(x, y_pred, color='k')
    plt.xlabel(r'$\log{T/K}$')
    plt.ylabel(r' Predicted Luminosity')
    plt.title(f'A graph of Temperature vs Luminosity with degree {i}')
    #plt.savefig(f'2Plot4_{i}.png', dpi=800)
    plt.close()

# importing the filtered data to be analysed
data2 = pd.read_csv('Q2b__stars.csv')

# defining each variable
T = data2['Temperature/K']
L = (data2['Luminosity(L/Lo)'])*(3.846e26)
R = data2['Radius(R/Ro)']

# creating a loop to obtain A
for i in range(len(R)):
    a = 4* pi *((R[i]*6.957e8)**2)
    A_list.append(a)

A = np.array(A_list)

# calculating the L/A
L_A = L/A

# defining a function to fit to
def fit_func(T, sigma):
    return sigma * (np.power(T, 4))

# creating a linspace to obtain a smoother curve
T_lin = np.linspace(T.min(), T.max(), 1000)

# using curve fit to obtain the best fitting curve
popt, pcov = curve_fit(fit_func, T, L_A)
fit_line = fit_func(T_lin, popt[0])

# plotting the graph
f = plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.title(r'A graph of $\frac{\mathrm{L}}{\mathrm{A}}$ against T')
plt.scatter(T, L_A, color='k')
plt.plot(T_lin, fit_line, color='k')
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.xlabel(r'T/K')
plt.ylabel(r'$\frac{\mathrm{L}}{\mathrm{A}}$ /Wm$^{-2}$')
plt.savefig('2Plot5.png', dpi=800)

# theoretical boltzmann constant
sigma_theoretical = 5.6696e-8 

boltz_accu = ((popt[0]/sigma_theoretical)-1)*100

# displaying the boltzamnn constant
print(f'The Boltzmann constant is: {popt[0]:.2E}, with a precision of {boltz_accu}')

# importing the third set of data
table_2_data = pd.read_excel('Q2c__Table_2_Data.xlsx')

L_data = (table_2_data['L/L0'])*(3.846e26)    
T_data = table_2_data['T/K']                              

# calculating the theoretical radii
r_theoretical = np.sqrt((L_data)/((4)*(pi)*(sigma_theoretical)*(T_data**4)))
print(f'Theoretical stellar radius: {r_theoretical}')

# calculating the experimental radii
r_experimental = np.sqrt((L_data)/((4)*(pi)*(popt[0])*(T_data**4)))
print(f'Experimental stellar radius: {r_experimental}')

for i in range(len(r_experimental)):
    r_accuracy = abs((r_experimental[i]/r_theoretical[i])-1)*100
    print(r_accuracy)
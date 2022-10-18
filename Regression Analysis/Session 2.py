import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from math import pi

A_list=[]

data = pd.read_csv('Q2a__HR_Diagram.csv')

data.groupby(['Star type'])

plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.tight_layout()

plt.scatter((data['Temperature/K']), data['Luminosity(L/Lo)'], color='k')
#plt.savefig('2Plot1.png', dpi=800)
plt.close()

plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.tight_layout()

plt.scatter(np.log(data['Temperature/K']), np.log(data['Luminosity(L/Lo)']), color='k')
#plt.savefig('2Plot2.png', dpi=800)
plt.close()

plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.tight_layout()

value_3 = data.mask(data['Star type']!=3).dropna().reset_index()
plt.scatter(np.log(value_3['Temperature/K']), np.log(value_3['Luminosity(L/Lo)']), color='k')
#plt.savefig('2Plot3.png', dpi=800)
plt.close()

y = np.log(value_3['Luminosity(L/Lo)'])
x = np.log(value_3['Temperature/K'])

x_a = x.array.reshape(-1,1)
poly = pf(degree=30)
poly_Lumen=poly.fit_transform(x_a)

model = LinearRegression()
model.fit(poly_Lumen, y)
y_pred = model.predict(poly_Lumen)

rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f'The root mean square is: {rmse}')
quit()

plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.tight_layout()

plt.scatter(x, y, color='k')
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x, y_pred), key=sort_axis)
x, y_pred = zip(*sorted_zip)
plt.plot(x, y_pred, color='k')
#plt.savefig('2Plot4.png', dpi=800)

plt.close()

data2 = pd.read_csv('Q2b__stars.csv')

T = data2['Temperature/K']
L = data2['Luminosity(L/Lo)']
R = data2['Radius(R/Ro)']

for i in range(len(R)):
    a = 4* pi *(R[i]**2)
    A_list.append(a)

A = np.array(A_list)

L_A = L/A

def fit_func(T, A, sigma):
    return sigma * (np.power(T, 4))

popt, pcov = curve_fit(fit_func, T, L_A, p0=(0,0))
fit_line = fit_func(T, popt[0], popt[1])

plt.scatter(T, L_A, color='k')
plt.plot(T, fit_line, color='k')
#plt.show()

print(f'Sigma is: {popt[1]}')
print(popt[0])

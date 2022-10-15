import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('Q2__HR_Diagram.csv')

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
plt.savefig('2Plot1.png', dpi=800)
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
plt.savefig('2Plot2.png', dpi=800)
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
plt.savefig('2Plot3.png', dpi=800)
plt.close()

y = np.log(value_3['Luminosity(L/Lo)'])
x = np.log(value_3['Temperature/K'])

x_a = x.array.reshape(-1,1)
poly = pf(degree=20)
poly_Lumen=poly.fit_transform(x_a)

model = LinearRegression()
model.fit(poly_Lumen, y)
y_pred = model.predict(poly_Lumen)

rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f'The root mean square is: {rmse}')

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
plt.savefig('2Plot4.png', dpi=800)
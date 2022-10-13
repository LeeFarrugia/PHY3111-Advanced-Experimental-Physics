import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

data = pd.read_csv('Q2__HR_Diagram.csv')

data.groupby(['Star type'])

plt.scatter(data['Luminosity(L/Lo)'], (data['Temperature/K']))
plt.savefig('2Plot1.png', dpi=800)
plt.close()

plt.scatter(np.log(data['Luminosity(L/Lo)']), np.log(data['Temperature/K']))
plt.savefig('2Plot2.png', dpi=800)

for i in range(len(data['Luminosity(L/Lo)'])):
    if data['Star type'][i] == 3:
        print('hobbok eric')
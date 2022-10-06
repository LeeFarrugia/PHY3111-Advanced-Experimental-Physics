import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

xlist = []
ylist = []
prodlist = []
denomlist = []
ydenomlist = []
yelist = []

data = pd.read_csv('Q1__Youngs_Modulus_of_a_Wire.csv')
diameter_array = (data['Diameter/m']).to_numpy()
mass_array = data['m/kg'].to_numpy()
x1_array = data['x_1/m'].to_numpy()
x2_array = data['x_2/m'].to_numpy()
x3_array = data['x_3/m'].to_numpy()
x4_array = data['x_4/m'].to_numpy()
L_array = data['L/m'].to_numpy()
x0_array = data['x_0/m'].to_numpy()

for i in range(len(x1_array)-1):
    x = sum([x1_array[i], x2_array[i], x3_array[i], x4_array[i]])
    X = (x/4)-x0_array[0]
    xlist.append(abs(X))

xi = np.array(xlist)**2
xbar = np.mean(xi)
print(xbar)

for i in range(len(mass_array)-1):
    yi = mass_array[i]/xlist[i]
    ylist.append(yi)

yi = np.array(ylist)
ybar = np.mean(yi)
print(ybar)

def beta_alpha_function(xi, xbar, yi, ybar):
    for i in range(len(xi)):
        prod = (xi[i]-xbar)*(yi[i]-ybar)
        prodlist.append(prod)
        xdenom = (xi[i]-xbar)**2
        denomlist.append(xdenom)
        ydenom = (yi[i]-ybar)**2
        ydenomlist.append(ydenom)
    prod_array = np.array(prodlist)
    denom_array = np.array(denomlist)
    ydenom_array = np.array(ydenomlist)
    numerator = sum(prod_array)
    denominator = sum(denom_array)
    ydenominator = sum(ydenom_array)
    beta = numerator/denominator
    alpha = ybar - (beta*xbar)
    r = numerator/(np.sqrt(denominator*ydenominator))
    delta_beta = (beta/(np.sqrt(len(xi)-2)))*((1/r**2)-1)
    delta_alpha = delta_beta*((1/len(xi))*(sum(xi**2)))
    R = r**2
    return beta, alpha, delta_beta, delta_alpha, r, R


beta, alpha, delta_beta, delta_alpha, r, R = beta_alpha_function(xi, xbar, yi, ybar)

for i in range(len(xi)):
    ye = alpha + (beta*xi[i])
    yelist.append(ye)
ye_array = np.array(yelist)

coeffs, cov = np.polyfit(xi, ye_array, 1, cov=True)
polyfunc = np.poly1d(coeffs)
trendline = polyfunc(xi)
print(ye_array, trendline)

plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.plot(xi, trendline)
plt.scatter(xi, ye_array)

print(coeffs[0])
plt.show()
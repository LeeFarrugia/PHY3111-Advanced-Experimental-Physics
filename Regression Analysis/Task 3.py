# Task 3

from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# importing the data to be analysed
data = pd.read_csv('Q3__Isotope_Decay_Dataset.csv')

# initialising the lists to be used
index_list=[]
slice_list=[]
lst = []
half_life_list = []
z_list = []
n_list = []
new_list =[]
new_z_list = []
new_n_list = []

# defining array to be used
even_a = np.arange(0,5922,2)
odd_a = np.arange(1,5921,2)

# importing specific data for A
a_values = data['A']
# creating an index with the data selected
index_array = np.arange(0, len(a_values)+20, 20)

# finding the ean values of A for each isotope, 20 values each
mean_a = [np.mean(a_values.iloc[index_array[i]:index_array[i+1]]) for i in range(len(index_array)-1)]

# finding which values of the mean are lower than 95 to account for the noise of the values and storing their index
for i in range(len(mean_a)):
    if mean_a[i] < 95:
        index_list.append(i)

# creating new list to slice the data according to the indices from before
for i in range(len(index_list)):
    slice_list.append((index_list[i])*20)
    slice_list.append((index_list[i]+1)*20)

# keeping only the data for the unstable isotopes
data_sliced = [data.iloc[slice_list[even_a[i]]:slice_list[odd_a[i]]] for i in range(len(even_a)-1)]

# removing any empty values
data_sliced = list(filter(lambda df: not df.empty, data_sliced))

# defining an empty dataframe
df = pd.DataFrame(columns=['z','n','t/s','A'])

# creating a dataframe with the sliced values for later use
for i in range(len(data_sliced)):
    temp_df = pd.DataFrame(data_sliced[i], columns=['z', 'n', 't/s', 'A'])
    df = pd.concat([df,temp_df]).reset_index(drop=True)

# indexing the new dataframe
index_df = np.arange(0, len(df), 1)
df.reset_index(drop=True).set_index(index_df, inplace=True)

# selecting only the data for A and t
a_uvalues = df['A']
t_values = df['t/s']

# finding the mean for A and t for each isotope
mean_ua = [np.mean(a_uvalues.iloc[index_array[i]:index_array[i+1]]) for i in range(len(index_array)-1)]
mean_t = [np.mean(t_values.iloc[index_array[i]:index_array[i+1]]) for i in range(len(index_array)-1)]

# edfinign the plotting parameters
plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.tight_layout()

# plotting the data
plt.scatter(np.log(mean_t), np.log(mean_ua), color='k')
plt.savefig('3Plot1.png', dpi=800)
plt.close()

# selecting the data only for calcium
calcium_df = df[df['z']==20]
# selecting only the data for 1 calcium isotope
calcium_df = df.iloc[5680:6080]
# re-indexing
calcium_df.reset_index(drop=True, inplace=True)

# defining the values and data to be used for calcium
calcium_a = calcium_df['A'][0:20]
calcium_log_a = np.log(calcium_df['A'][0:20])
calcium_t = (calcium_df['t/s'][0:20])

# defining the function to find the value of A/A0
def fit_func(t, thalf):
    return (np.exp((-1 * t * np.log(2))/thalf))

# using curve fit to calculate the value of the half life
popt, pcov = curve_fit(fit_func, calcium_t, (calcium_a/calcium_a[0]))

# obtaining the curve of the calclium isotope decay
fitted_line = fit_func(calcium_t, popt[0])
print(f'The half life of calcium-14 is: {popt[0]:.2E}s')

# obtaining the straight line to compare values obtained
coeffs, cov = np.polyfit(calcium_t, calcium_log_a, 1, cov=True)
polyfunc = np.poly1d(coeffs)
trendline = polyfunc(calcium_t)

print(f'The half life of calcium-14 is from the srtaight line graph: {-np.log(2)/coeffs[0]:.2E}s')

# defining the subplots
f, (a0, a1) = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(7.3, 10.7))

# defining the font to be used
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

# plotting the curve
a0.scatter(calcium_t, (calcium_a/calcium_a[0]), color='k', label='Data Points')
a0.plot(calcium_t, fitted_line, color='k', label='Trendline')
a0.minorticks_on()
a0.grid(visible=True, which='major', linestyle='-')
a0.grid(visible=True, which='minor', linestyle='--')
a0.set_xlabel('Time/s')
a0.set_ylabel('A/A0')
a0.set_title(r'A graph $\frac{A}{A_0}$ vs Time')

# plotting the straight line
a1.scatter(calcium_t, calcium_log_a, color='k', label='Data Point')
a1.plot(calcium_t, trendline, color='k', label='Trendline')
a1.minorticks_on()
a1.grid(visible=True, which='major', linestyle='-')
a1.grid(visible=True, which='minor', linestyle='--')
a1.set_xlabel('Time/s')
a1.set_ylabel(r'$log(A)$')
a1.set_title(r'A graph of $log(A)$ vs Time')

# removing the excess space, showing legend and saving figure
f.tight_layout()
f.legend()
f.savefig('3Plot2.png', dpi=800)
plt.close()

# defining specific data columns
df_t = df['t/s']
df_a = df['A']

# creating array for selections of data
selection_array = np.arange(0, len(df['t/s']), 20)

# applying the curve fit function on all of the data
for i in range(len(selection_array)-1):
    da = df_a[selection_array[i]:selection_array[i+1]]
    dt = df_t[selection_array[i]:selection_array[i+1]]
    da0 = df_a[selection_array[i]]
    da_da0 = da/da0
    popt, pcov = curve_fit(fit_func, dt, da_da0)
    half_life_list.append(popt[0])
    z_list.append(df['z'][selection_array[i]])
    n_list.append(df['n'][selection_array[i]])

# joining the 3 lists together
plotting_data = list(zip(z_list, n_list, half_life_list))

# creating a data frame for Z, N, Half life
results_df = pd.DataFrame({'Z':z_list, 'N': n_list, 'Half Life/s': half_life_list})
results_df.to_excel('Table.xlsx')
plot_data = results_df.pivot(index='Z', columns='N',
values='Half Life/s')

# defining the font to be used
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

# plotting the Heat map
ax = sns.heatmap(plot_data, square=True, norm=LogNorm())
ax.invert_yaxis()

# finding the index for when Z = N
for i in range(len(results_df)):
    if results_df['Z'][i] == results_df['N'][i]:
        new_list.append(i)

# selecting the data for when Z = N
for i in new_list:
    new_z_list.append(z_list[i])
    new_n_list.append(n_list[i])

# changing the data from before of Z = N, into a dataframe
z_n = pd.DataFrame({'Z': new_z_list, 'N': new_n_list}) 

# plotting the straight line for the data
plt.plot(z_n['N'], z_n['Z'], color='k')

# saving the figure
plt.savefig('3Plot3.png', dpi=800)
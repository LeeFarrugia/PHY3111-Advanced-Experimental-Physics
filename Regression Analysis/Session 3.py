from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

data = pd.read_csv('Q3__Isotope_Decay_Dataset.csv')

index_list=[]
slice_list=[]
lst = []
half_life_list = []
z_list = []
n_list = []
new_list =[]
new_z_list = []
new_n_list = []

even_a = np.arange(0,5922,2)
odd_a = np.arange(1,5921,2)

a_values = data['A']
index_array = np.arange(0, len(a_values)+20, 20)

mean_a = [np.mean(a_values.iloc[index_array[i]:index_array[i+1]]) for i in range(len(index_array)-1)]

for i in range(len(mean_a)):
    if mean_a[i] < 95:
        index_list.append(i)

for i in range(len(index_list)):
    slice_list.append((index_list[i])*20)
    slice_list.append((index_list[i]+1)*20)

data_sliced = [data.iloc[slice_list[even_a[i]]:slice_list[odd_a[i]]] for i in range(len(even_a)-1)]

data_sliced = list(filter(lambda df: not df.empty, data_sliced))

df = pd.DataFrame(columns=['z','n','t/s','A'])

for i in range(len(data_sliced)):
    temp_df = pd.DataFrame(data_sliced[i], columns=['z', 'n', 't/s', 'A'])
    df = pd.concat([df,temp_df]).reset_index(drop=True)

index_df = np.arange(0, len(df), 1)
df.reset_index(drop=True).set_index(index_df, inplace=True)

a_uvalues = df['A']
t_values = df['t/s']

mean_ua = [np.mean(a_uvalues.iloc[index_array[i]:index_array[i+1]]) for i in range(len(index_array)-1)]
mean_t = [np.mean(t_values.iloc[index_array[i]:index_array[i+1]]) for i in range(len(index_array)-1)]

plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')
plt.tight_layout()

plt.scatter(np.log(mean_t), np.log(mean_ua), color='k')
#plt.savefig('3Plot1.png', dpi=800)
plt.close()

calcium_df = df[df['z']==20]
calcium_df = df.iloc[5680:6080]
calcium_df.reset_index(drop=True, inplace=True)

calcium_a = calcium_df['A'][0:20]
calcium_log_a = np.log(calcium_df['A'][0:20])
calcium_t = (calcium_df['t/s'][0:20])

def fit_func(t, thalf):
    return (np.exp((-1 * t * np.log(2))/thalf))

popt, pcov = curve_fit(fit_func, calcium_t, (calcium_a/calcium_a[0]))

fitted_line = fit_func(calcium_t, popt[0])
print(f'The half life of calcium-14 is: {popt[0]:.2E}s')

# obtaining the straight line to compare values obtained
coeffs, cov = np.polyfit(calcium_t, calcium_log_a, 1, cov=True)
polyfunc = np.poly1d(coeffs)
trendline = polyfunc(calcium_t)

print(f'The half life of calcium-14 is from the srtaight line graph: {-np.log(2)/coeffs[0]:.2E}s')

f, (a0, a1) = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(7.3, 10.7))

plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'

a0.scatter(calcium_t, (calcium_a/calcium_a[0]), color='k', label='Data Points')
a0.plot(calcium_t, fitted_line, color='k', label='Trendline')
a0.minorticks_on()
a0.grid(visible=True, which='major', linestyle='-')
a0.grid(visible=True, which='minor', linestyle='--')
a0.set_xlabel('Time/s')
a0.set_ylabel('A/A0')
a0.set_title(r'A graph $\frac{A}{A_0}$ vs Time')

a1.scatter(calcium_t, calcium_log_a, color='k', label='Data Point')
a1.plot(calcium_t, trendline, color='k', label='Trendline')
a1.minorticks_on()
a1.grid(visible=True, which='major', linestyle='-')
a1.grid(visible=True, which='minor', linestyle='--')
a1.set_xlabel('Time/s')
a1.set_ylabel(r'$log(A)$')
a1.set_title(r'A graph of $log(A)$ vs Time')

f.tight_layout()
f.legend()
#f.savefig('3Plot2.png', dpi=800)
plt.close()

df_t = df['t/s']
df_a = df['A']

selection_array = np.arange(0, len(df['t/s']), 20)

for i in range(len(selection_array)-1):
    da = df_a[selection_array[i]:selection_array[i+1]]
    dt = df_t[selection_array[i]:selection_array[i+1]]
    da0 = df_a[selection_array[i]]
    da_da0 = da/da0
    popt, pcov = curve_fit(fit_func, dt, da_da0)
    half_life_list.append(popt[0])
    z_list.append(df['z'][selection_array[i]])
    n_list.append(df['n'][selection_array[i]])

plotting_data = list(zip(z_list, n_list, half_life_list))

results_df = pd.DataFrame({'Z':z_list, 'N': n_list, 'Half Life/s': half_life_list})
plot_data = results_df.pivot(index='Z', columns='N',
values='Half Life/s')

ax = sns.heatmap(plot_data, square=True, norm=LogNorm())
ax.invert_yaxis()

for i in range(len(results_df)):
    if results_df['Z'][i] == results_df['N'][i]:
        new_list.append(i)

print(new_list)

for i in new_list:
    new_z_list.append(z_list[i])
    new_n_list.append(n_list[i])

z_n = pd.DataFrame({'Z': new_z_list, 'N': new_n_list}) 
plt.plot(z_n['N'], z_n['Z'], color='k')
print(z_n)
plt.show()
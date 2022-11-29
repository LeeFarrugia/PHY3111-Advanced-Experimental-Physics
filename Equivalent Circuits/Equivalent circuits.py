import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing data to different dataframes for each sheet
data_air = pd.read_excel('Data.xlsx', 0)
data_short = pd.read_excel('Data.xlsx', 1)
data_water = pd.read_excel('Data.xlsx', 2)
data_methanol = pd.read_excel('Data.xlsx', 3).dropna()
data_NaCl = pd.read_excel('Data.xlsx', 4).dropna()
e_methanol = pd.read_excel('Data.xlsx', 5).dropna()
e_NaCl = pd.read_excel('Data.xlsx', 6).dropna()

# defining the frequency variable
frequency = np.array(data_air['%freq[Hz]'])

# finding the average of the real part of air
air_real_avg = data_air[['Real (S11) R1', 'Real (S11) R2', 'Real (S11) R3', 'Real (S11) R4' , 'Real (S11) R5', 'Real (S11) R6']].mean(axis=1)
data_air['Real Avg'] = air_real_avg

# finding the average of the imaginary part of air
air_imaginary_avg = data_air[['Imag (S11) R1', 'Imag (S11) R2', 'Imag (S11) R3', 'Imag (S11) R4' , 'Imag (S11) R5', 'Imag (S11) R6']].mean(axis=1)
data_air['Imag Avg'] = air_imaginary_avg

# finding the average of the real part of water
water_real_avg = data_water[['Real (S11) R1', 'Real (S11) R2', 'Real (S11) R3', 'Real (S11) R4' , 'Real (S11) R5', 'Real (S11) R6']].mean(axis=1)
data_water['Real Avg'] = water_real_avg

# finding the average of the imaginary part of water
water_imaginary_avg = data_water[['Imag (S11) R1', 'Imag (S11) R2', 'Imag (S11) R3', 'Imag (S11) R4' , 'Imag (S11) R5', 'Imag (S11) R6']].mean(axis=1)
data_water['Imag Avg'] = water_imaginary_avg

# finding the average of the real part of methanol
methanol_real_avg = data_methanol[['Real (S11) R1', 'Real (S11) R2', 'Real (S11) R3', 'Real (S11) R4' , 'Real (S11) R5', 'Real (S11) R6']].mean(axis=1)
data_methanol['Real Avg'] = methanol_real_avg

# finding the average of the imaginary part of methanol
methanol_imaginary_avg = data_methanol[['Imag (S11) R1', 'Imag (S11) R2', 'Imag (S11) R3', 'Imag (S11) R4' , 'Imag (S11) R5', 'Imag (S11) R6']].mean(axis=1)
data_methanol['Imag Avg'] = methanol_imaginary_avg

# finding the average of the real part of NaCl
NaCl_real_avg = data_NaCl[['Real (S11) R1', 'Real (S11) R2', 'Real (S11) R3', 'Real (S11) R4' , 'Real (S11) R5', 'Real (S11) R6']].mean(axis=1)
data_NaCl['Real Avg'] = NaCl_real_avg

# finding the average of the imaginary part of NaCl
NaCl_imaginary_avg = data_NaCl[['Imag (S11) R1', 'Imag (S11) R2', 'Imag (S11) R3', 'Imag (S11) R4' , 'Imag (S11) R5', 'Imag (S11) R6']].mean(axis=1)
data_NaCl['Imag Avg'] = NaCl_imaginary_avg

# finding the average of the given real permittivity of methanol
e_methanol_real_avg = e_methanol[["e'_R1","e'_R2","e'_R3"]].mean(axis=1)
e_methanol['Real Avg'] = e_methanol_real_avg

# finding the average of the given imaginary permittivity of methanol
e_methanol_imaginary_avg = e_methanol[["e''_R1","e''_R2","e''_R3"]].mean(axis=1)
e_methanol['Imag Avg'] = e_methanol_imaginary_avg

# finding the average of the given real permittivity of NaCl
e_NaCl_real_avg = e_NaCl[["e'_R1","e'_R2","e'_R3"]].mean(axis=1)
e_NaCl['Real Avg'] = e_NaCl_real_avg

# finding the average of the given imaginary permittivity of NaCl
e_NaCl_imaginary_avg = e_NaCl[["e''_R1","e''_R2","e''_R3"]].mean(axis=1)
e_NaCl['Imag Avg'] = e_NaCl_imaginary_avg

# defining the different perimittivity as numpy array
short_r_avg = data_short['re:Trc1_S11'].to_numpy()
short_i_avg = data_short['im:Trc1_S11'].to_numpy()

air_r_avg = data_air['Real Avg'].to_numpy()
air_i_avg = data_air['Imag Avg'].to_numpy()

water_r_avg = data_water['Real Avg'].to_numpy()
water_i_avg = data_water['Imag Avg'].to_numpy()

methanol_r_avg = data_methanol['Real Avg'].to_numpy()
methanol_i_avg = data_methanol['Imag Avg'].to_numpy()

NaCl_r_avg = data_NaCl['Real Avg'].to_numpy()
NaCl_i_avg = data_NaCl['Imag Avg'].to_numpy()

e_methanol_r_avg = e_methanol['Real Avg'].to_numpy()
e_methanol_i_avg = e_methanol['Imag Avg'].to_numpy()

e_NaCl_r_avg = e_NaCl['Real Avg'].to_numpy()
e_NaCl_i_avg = e_NaCl['Imag Avg'].to_numpy()

# creating the complex numbers from the averages
air_complex = air_r_avg - (1j * air_i_avg)
short_complex = short_r_avg - (1j * short_i_avg)
water_complex = water_r_avg - (1j * water_i_avg)
methanol_complex = methanol_r_avg - (1j * methanol_i_avg)
NaCl_complex = NaCl_r_avg - (1j * NaCl_i_avg)
e_methanol_c = e_methanol_r_avg - (1j * e_methanol_i_avg)
e_NaCl_c = e_NaCl_r_avg - (1j * e_NaCl_i_avg)

# calculating the different deltas required in the equation
delta_13 = np.subtract(short_complex, water_complex)
delta_21 = np.subtract(air_complex, short_complex)
delta_23 = np.subtract(air_complex, water_complex)
delta_32 = np.subtract(water_complex, air_complex)
delta_m1_methanol = np.subtract(methanol_complex, short_complex)
delta_m2_methanol = np.subtract(methanol_complex, air_complex)
delta_m3_methanol = np.subtract(methanol_complex, water_complex)
delta_m1_NaCl = np.subtract(NaCl_complex, short_complex)
delta_m2_NaCl = np.subtract(NaCl_complex, air_complex)
delta_m3_NaCl = np.subtract(NaCl_complex, water_complex)

# using the equation and values to calculate the complex permittivity
e_methanol = -1 * (((delta_m2_methanol * delta_13) / (delta_m1_methanol * delta_32)) * 80.5) - (((delta_m3_methanol * delta_21) / (delta_m1_methanol * delta_32)) * 1)
e_NaCl = -1 * (((delta_m2_NaCl * delta_13) / (delta_m1_NaCl * delta_32)) * 80.5) - (((delta_m3_NaCl * delta_21) / (delta_m1_NaCl * delta_32)) * 1)

# taking the real parts only
methanol_real = np.real(e_methanol)
e_methanol_real = np.real(e_methanol_c)
NaCl_real = np.real(e_NaCl)
e_NaCl_real = np.real(e_NaCl_c)

# polyfit calculated methanol
coefficients, cov = np.polyfit(frequency, methanol_real, 5, cov=True)
poly_function = np.poly1d(coefficients)
trendline_c_m = poly_function(frequency)

# polyfit given methanol
coefficients, cov = np.polyfit(frequency, e_methanol_real, 5, cov=True)
poly_function = np.poly1d(coefficients)
trendline_g_m = poly_function(frequency)

# polyfit calculated NaCl
coefficients, cov = np.polyfit(frequency, NaCl_real, 5, cov=True)
poly_function = np.poly1d(coefficients)
trendline_c_n = poly_function(frequency)

# polyfit given NaCl
coefficients, cov = np.polyfit(frequency, e_NaCl_real, 5, cov=True)
poly_function = np.poly1d(coefficients)
trendline_g_n = poly_function(frequency)

# value comparisons
positions = (0, 74, 174, 200)
for i in positions:
    print(f'The calculated permittivity at data point {i+1} for methanol is:{e_methanol[i]:.2f}, while the given value is: {e_methanol_c[i]:.2f}')
    print(f'The calculated permittivity at data point {i+1} for NaCl is:{e_NaCl[i]:.2f}, while the given value is: {e_NaCl_c[i]:.2f}')

# plotting the data for methanol
plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')

plt.scatter(frequency, methanol_real, color='k', label='Calculated')
plt.plot(frequency, trendline_c_m, color = 'k', label='Calculated')
plt.scatter(frequency, e_methanol_real, color='grey', label='Given')
plt.plot(frequency, trendline_g_m, '--',  color = 'k', label = 'Given')
plt.xlabel('f/Hz')
plt.ylabel(r'$\mathrm{\epsilon_{methanol}}$')
plt.title('Complex Permittivity of Methanol vs Frequency')
plt.legend()
plt.savefig('Plot1.png', dpi=800)
plt.show()
plt.close()

# plotting the data of NaCl
plt.figure(figsize=(7.5, 10.5))
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
plt.minorticks_on()
plt.grid(visible=True, which='major', linestyle='-')
plt.grid(visible=True, which='minor', linestyle='--')

plt.scatter(frequency, NaCl_real, color='k', label='Calculated')
plt.plot(frequency, trendline_c_n, color = 'k', label='Calculated')
plt.scatter(frequency, e_NaCl_real, color='grey', label='Given')
plt.plot(frequency, trendline_g_n, '--',  color = 'k', label = 'Given')
plt.xlabel('f/Hz')
plt.ylabel(r'$\mathrm{\epsilon_{NaCl}}$')
plt.title('Complex Permittivity of NaCl vs Frequency')
plt.legend()
plt.savefig('Plot2.png', dpi=800)
plt.show()

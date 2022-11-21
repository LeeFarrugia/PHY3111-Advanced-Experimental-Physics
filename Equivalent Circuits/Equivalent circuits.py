import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_air = pd.read_excel('Data.xlsx', 0)
data_short = pd.read_excel('Data.xlsx', 1)
data_water = pd.read_excel('Data.xlsx', 2)
data_methanol = pd.read_excel('Data.xlsx', 3).dropna()
data_NaCl = pd.read_excel('Data.xlsx', 4).dropna()
e_methanol = pd.read_excel('Data.xlsx', 5).dropna()
e_NaCl = pd.read_excel('Data.xlsx', 6).dropna()

air_real =[]
air_imaginary = []
short_real = data_short['re:Trc1_S11']
short_imaginary = data_short['im:Trc1_S11']
water_real = []
water_imaginary = []
methanol_real = []
methanol_imaginary = []
NaCl_real = []
NaCl_imaginary = []
e_methanol_real = []
e_methanol_imaginary = []
e_NaCl_real = []
e_NaCl_imaginary = []

for i in range(len(data_air['Real (S11) R1'])):
    air_r_avg = np.average([data_air['Real (S11) R1'][i], data_air['Real (S11) R2'][i], data_air['Real (S11) R3'][i], data_air['Real (S11) R4'][i], data_air['Real (S11) R5'][i], data_air['Real (S11) R6'][i]])
    air_real.append(float(air_r_avg))

    air_i_avg = np.average([data_air['Imag (S11) R1'][i], data_air['Imag (S11) R2'][i], data_air['Imag (S11) R3'][i], data_air['Imag (S11) R4'][i], data_air['Imag (S11) R5'][i], data_air['Imag (S11) R6'][i]])
    air_imaginary.append(float(air_i_avg))

    water_r_avg = np.average([data_water['Real (S11) R1'][i], data_water['Real (S11) R2'][i], data_water['Real (S11) R3'][i], data_water['Real (S11) R4'][i], data_water['Real (S11) R5'][i], data_water['Real (S11) R6'][i]])
    water_real.append(water_r_avg)

    water_i_avg = np.average([data_water['Imag (S11) R1'][i], data_water['Imag (S11) R2'][i], data_water['Imag (S11) R3'][i], data_water['Imag (S11) R4'][i], data_water['Imag (S11) R5'][i], data_water['Imag (S11) R6'][i]])
    water_imaginary.append(water_i_avg)

    methanol_r_avg = np.average([data_methanol['Real (S11) R1'][i], data_methanol['Real (S11) R2'][i], data_methanol['Real (S11) R3'][i], data_methanol['Real (S11) R4'][i], data_methanol['Real (S11) R5'][i], data_methanol['Real (S11) R6'][i]])
    methanol_real.append(methanol_r_avg)

    methanol_i_avg = np.average([data_methanol['Imag (S11) R1'][i], data_methanol['Imag (S11) R2'][i], data_methanol['Imag (S11) R3'][i], data_methanol['Imag (S11) R4'][i], data_methanol['Imag (S11) R5'][i], data_methanol['Imag (S11) R6'][i]])
    methanol_imaginary.append(methanol_i_avg)

    NaCl_r_avg = np.average([data_NaCl['Real (S11) R1'][i], data_NaCl['Real (S11) R2'][i], data_NaCl['Real (S11) R3'][i], data_NaCl['Real (S11) R4'][i], data_NaCl['Real (S11) R5'][i], data_NaCl['Real (S11) R6'][i]])
    NaCl_real.append(NaCl_r_avg)

    NaCl_i_avg = np.average([data_NaCl['Imag (S11) R1'][i], data_NaCl['Imag (S11) R2'][i], data_NaCl['Imag (S11) R3'][i], data_NaCl['Imag (S11) R4'][i], data_NaCl['Imag (S11) R5'][i], data_NaCl['Imag (S11) R6'][i]])
    NaCl_imaginary.append(NaCl_i_avg)

    e_methanol_r_avg = np.average([e_methanol["e'_R1"][i], e_methanol["e'_R2"][i], e_methanol["e'_R3"][i]])
    e_methanol_real.append(e_methanol_r_avg)

    e_methanol_i_avg = np.average([e_methanol["e''_R1"][i], e_methanol["e''_R2"][i], e_methanol["e''_R3"][i]])
    e_methanol_imaginary.append(e_methanol_i_avg)

    e_NaCl_r_avg = np.average([e_NaCl["e'_R1"][i], e_NaCl["e'_R2"][i], e_NaCl["e'_R3"][i]])
    e_NaCl_real.append(e_NaCl_r_avg)

    e_NaCl_i_avg = np.average([e_NaCl["e''_R1"][i], e_NaCl["e''_R2"][i], e_NaCl["e''_R3"][i]])
    e_NaCl_imaginary.append(e_NaCl_i_avg)

print('done')

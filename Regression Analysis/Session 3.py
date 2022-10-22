import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

data = pd.read_csv('Q3__Isotope_Decay_Dataset.csv')

index_list=[]
slice_list=[]
lst = []
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

print(index_list[0:10])
print(slice_list[0:10])
print(len(slice_list))

data_sliced = list(filter(lambda df: not df.empty, data_sliced))

df = pd.DataFrame(columns=['z','n','t/s','A'])

for i in range(len(data_sliced)):
    temp_df = pd.DataFrame(data_sliced[i], columns=['z', 'n', 't/s', 'A'])
    df = pd.concat([df,temp_df])

df.to_clipboard()
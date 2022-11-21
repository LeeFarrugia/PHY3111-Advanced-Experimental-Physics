import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_air = pd.read_excel('Data.xlsx', 0)
data_air.rename(columns=data_air.iloc[1], inplace=True)  # type: ignore
data_air = data_air.iloc[2:]
data_short = pd.read_excel('Data.xlsx', 1)
data_short.rename(columns=data_short.iloc[1], inplace=True)  # type: ignore
data_short = data_short.iloc[2:]
data_water = pd.read_excel('Data.xlsx', 2)
data_water.rename(columns=data_water.iloc[1], inplace=True)  # type: ignore
data_water = data_water.iloc[2:]
data_methanol = pd.read_excel('Data.xlsx', 3).dropna()
data_NaCl = pd.read_excel('Data.xlsx', 4).dropna()
e_methanol = pd.read_excel('Data.xlsx', 5).dropna()
e_NaCl = pd.read_excel('Data.xlsx', 6).dropna()


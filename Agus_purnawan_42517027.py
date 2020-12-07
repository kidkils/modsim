import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

url = 'https://raw.githubusercontent.com/kidkils/modsim/main/chinaGDP.txt'
df = pd.read_csv(url)

plt.figure(figsize=(8,5))
x_data, y_data = (df['Year'].values, df['Value'].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

def sigmoid(x, beta1, beta2):
  return 1/(1+np.exp(-beta1*(x-beta2)))

# normalisasi x
xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

# menentukan nilai parameter
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
print(popt)

# normalisasi
x = x_data/max(x_data)
# fit
y = sigmoid(x, *popt)

plt.figure(figsize=(8, 5))
plt.plot(x_data, ydata, 'ro', label='data')
plt.plot(x_data, y, label='fit')
plt.legend(loc='best')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()

# generate angka tahun dari 2015 - 2030
x_2015_2030 = np.linspace(2015, 2030, num=16)
# normalisasi x_2015_2030
x_2015_2030_norm = x_2015_2030/max(x_data)
# fit
y = sigmoid(x_2015_2030_norm, *popt)

plt.figure(figsize=(8, 5))
plt.plot(x_2015_2030, y, label='prediksi')
plt.legend(loc='best')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()

# data prediksi
y = sigmoid(xdata, *popt)

# MAE
print(mean_absolute_error(ydata, y))

# MSE
print(mean_squared_error(ydata, y))

# R2
print(r2_score(ydata, y))

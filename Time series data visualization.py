# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 00:06:00 2020

@author: Abhimanyu Trakroo
"""

# Importing Relevant Packages
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.graphics.tsaplots as sgt 
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
sns.set()

### Loading and Transforming the Data

raw_csv_data = pd.read_csv("..../Index2018.csv") 
df_comp=raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index("date", inplace=True)
df_comp=df_comp.asfreq('b')
df_comp=df_comp.fillna(method='ffill')



### Removing Surplus Data
### Removing Surplus Data

df_comp['market_value']=df_comp.spx
del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

### White Noise
### White Noise

wn = np.random.normal(loc = df.market_value.mean(), scale = df.market_value.std(), size = len(df))
df['wn'] = wn
df.describe()

df.wn.plot(figsize = (20,5))
plt.title("White Noise Time-Series", size= 24)
plt.show()

df.market_value.plot(figsize=(20,5))
plt.title("S&P Prices", size = 24)
plt.ylim(0,2300)
plt.show()

### Random Walk
### Random Walk
### Random Walk

### Generating Random Walk
### Creating a Function to Generate Random Walks

def rw_gen(T = 1, N = 100, mu = 0.1, sigma = 0.01, S0 = 20):        
    dt = float(T)/N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N) # Spits out a normal distribution
    W.max()   
    pd.DataFrame(W).plot()
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    pd.DataFrame(W).plot()
    X = (mu-0.5*sigma**2)*t + sigma*W 
    S = S0*np.exp(X) ### geometric brownian motion ###
    return S

 np.linspace(1,100,10) # Generates a series with steps

### Importing the Data Set for generating random walk
###### This step is not mandatory if you have a starting value in mind
    
############################################################################

raw_csv_data = pd.read_csv("Index2018.csv") 
df_comp=raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index("date", inplace=True)
df_comp=df_comp.asfreq('b')
df_comp=df_comp.fillna(method='ffill')
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]

### Setting the Parameters and Calling the Function
dates = pd.date_range('1994-01-07', '2013-04-05')                   # Same as the training set
T = (dates.max()-dates.min()).days / 365                            # We're generating daily values
N = dates.size                                                      # Number of observations
start_price = df.ftse.mean()                                        # We're using the mean of the our existing time series as a starting point
y = pd.Series(rw_gen(T, N, sigma=0.3, S0=start_price), index=dates) # Calling the RW-generating function

#### Plotting the RW 
#
####### This is also optional. We're just checking what our new series looks like.
y.plot()
plt.show()

### Storing the Random Walk in a CSV file
 y.to_csv('"................/RandWalk2.csv', header = True) # Make sure to avoid using RandWalk.csv, since that will just write over the existing file
############ Generated the Random Walk ####################

############ Import the Random Walk ####################
rw = pd.read_csv("...../RandWalk.csv")
rw.date = pd.to_datetime(rw.date, dayfirst = True)
rw.set_index("date", inplace = True)
rw = rw.asfreq('b')

rw.describe()

df['rw'] = rw.price

df.head()

df.rw.plot(figsize = (20,5))
df.market_value.plot()
plt.title("Random Walk vs S&P", size = 24)
plt.show()

### Stationarity
### Stationarity
### Stationarity

sts.adfuller(df.market_value)

sts.adfuller(df.wn)

sts.adfuller(df.rw)

### Seasonality
### Seasonality
### Seasonality
### Seasonality

# Just a demo
#s_dec_additive = seasonal_decompose(df.market_value, model = "additive")
#s_dec_additive.plot()
#plt.show()


s_dec_multiplicative = seasonal_decompose(df.market_value, model = "multiplicative")
s_dec_multiplicative.plot()
plt.show()

### ACF
### ACF
### ACF
### ACF

sgt.plot_acf(df.market_value, lags = 40, zero = False)
plt.title("ACF S&P", size = 24)
plt.show()

sgt.plot_acf(df.wn, lags = 40, zero = False)
plt.title("ACF WN", size = 24)
plt.show()

sgt.plot_acf(df.rw, lags = 40, zero = False)
plt.title("ACF RW", size = 24)
plt.show()





### PACF
### PACF
### PACF
### PACF

sgt.plot_pacf(df.market_value, lags = 40, zero = False, method = ('ols'))
plt.title("PACF S&P", size = 24)
plt.show()


sgt.plot_pacf(df.wn, lags = 40, zero = False, method = ('ols'))
plt.title("PACF WN", size = 24)
plt.show()

sgt.plot_pacf(df.rw, lags = 40, zero = False, method = ('ols'))
plt.title("PACF RW", size = 24)
plt.show()

#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pmdarima.arima import auto_arima
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


# In[2]:


df_weather_1hr = pd.read_csv("weather_data_1hr.csv")


# In[3]:


df_weather_24hr = pd.read_csv("weather_data_24hr.csv")


# In[4]:


# df_weather_1hr.head(10)


# In[5]:


# df_weather_24hr.head(10)


# In[6]:


# df_weather_1hr.columns


# In[7]:


# df_weather_24hr.columns


# In[8]:


# df_weather_1hr.isna().sum()


# In[9]:


# df_weather_24hr.isna().sum()


# # Focusing on hourly data - df_weather_1hr

# In[10]:


# time axis is hourly time, where as the target variable is precipitationMM


# In[11]:


df = df_weather_1hr[["date", "time", "precipMM"]]
# df


# In[12]:


# converting the time to hourly time axis

# df.head(25)


# In[13]:


# df.info()


# In[14]:


keys = list(range(0, 2400, 100))
values = ['0' + str(x) + ':00:00' for x in range(0, 10, 1)] + \
    [str(x) + ':00:00' for x in range(10, 24, 1)]
time_conversion = dict(zip(keys, values))


def hourly_timestamp_conversion(row):
    return time_conversion[row["time"]]


# In[15]:


# hourly_timestamp_conversion(100)


# In[16]:


df['hourly_data'] = df.apply(hourly_timestamp_conversion, axis=1)


# In[17]:


df['time_stamp'] = df['date'] + " " + df['hourly_data']


# In[18]:


df.drop(["date", "time", "hourly_data"], axis=1, inplace=True)


# In[19]:


# df.info()


# In[20]:


df['datetime'] = pd.to_datetime(df['time_stamp'])
df = df.set_index('datetime')
df.drop(['time_stamp'], axis=1, inplace=True)
df.head()


# In[21]:


# save the checkpoint data for reuse

df.to_csv("cleaned_data.csv")


# Time Series have several key features such as trend, seasonality, and noise.
#
# ARIMA model, which stands for AutoRegressive Integrated Moving Average.
#
# In an ARIMA model there are 3 parameters that are used to help model the major aspects of a times series: seasonality, trend, and noise. These parameters are labeled p,d,and q.
#
# p is the parameter associated with the auto-regressive aspect of the model, which incorporates past values. For example, forecasting that if it rained a lot over the past few days, you state its likely that it will rain tomorrow as well.
#
# d is the parameter associated with the integrated part of the model, which effects the amount of differencing to apply to a time series. You can imagine an example of this as forecasting that the amount of rain tomorrow will be similar to the amount of rain today, if the daily amounts of rain have been similar over the past few days.
#
# q is the parameter associated with the moving average part of the model.
#
# If our model has a seasonal component, we use a seasonal ARIMA model (SARIMA). In that case we have another set of parameters: P,D, and Q which describe the same associations as p,d, and q, but correspond with the seasonal components of the model.

# In[22]:


df.shape


# In[23]:


df.columns = ['precipMM']


# In[24]:


#get_ipython().run_line_magic('matplotlib', 'inline')
df.plot()


# In[27]:


# take the last 10,000 rows as the data is humungous


# In[30]:


df_new = df.tail(5000)


# In[31]:


df_new.head()


# In[33]:


#get_ipython().run_line_magic('matplotlib', 'inline')
df_new.plot()


# In[34]:


# from statsmodels.tsa.seasonal import seasonal_decompose
# import matplotlib.pyplot as plt
# plt.figure(figsize=(25, 20))
# result = seasonal_decompose(df, model='additve')
# fig = result.plot()


# In[35]:


stepwise_model = auto_arima(df_new, start_p=1, start_q=1,
                            max_p=3,
                            max_q=3,
                            m=12,
                            start_P=0,
                            seasonal=True,
                            d=1, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)
print(stepwise_model.aic())


# In[ ]:


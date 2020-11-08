#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
#get_ipython().system('python --version')


# In[2]:


#get_ipython().system('pip3 -V')


# In[3]:


#get_ipython().system('pip3 list')


# In[4]:


print(sorted(sys.path))


# In[11]:


#get_ipython().system('ls "/Users/manishanker.talusani/Documents/work/karlos/10 april 2020/.venv_karlos_3.7.5/lib/python3.7/site-packages/pmdarima"')


# In[12]:


# In[27]:


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


# In[28]:


df_weather_1hr = pd.read_csv("weather_data_1hr.csv")


# In[29]:


df_weather_24hr = pd.read_csv("weather_data_24hr.csv")


# In[30]:


# df_weather_1hr.head(10)


# In[31]:


# df_weather_24hr.head(10)


# In[32]:


# df_weather_1hr.columns


# In[33]:


# df_weather_24hr.columns


# In[34]:


# df_weather_1hr.isna().sum()


# In[35]:


# df_weather_24hr.isna().sum()


# # Focusing on hourly data - df_weather_1hr

# In[36]:


# time axis is hourly time, where as the target variable is precipitationMM


# In[37]:


df = df_weather_1hr[["date", "time", "precipMM"]]
df


# In[38]:


# converting the time to hourly time axis

# df.head(25)


# In[39]:


# df.info()

# In[40]:


keys = list(range(0, 2400, 100))
values = ['0' + str(x) + ':00:00' for x in range(0, 10, 1)] + \
    [str(x) + ':00:00' for x in range(10, 24, 1)]
time_conversion = dict(zip(keys, values))


def hourly_timestamp_conversion(row):
    return time_conversion[row["time"]]


# In[41]:


# hourly_timestamp_conversion(100)


# In[42]:


df['hourly_data'] = df.apply(hourly_timestamp_conversion, axis=1)


# In[43]:


df['time_stamp'] = df['date'] + " " + df['hourly_data']


# In[44]:


df.drop(["date", "time", "hourly_data"], axis=1, inplace=True)


# In[19]:


# df.info()


# In[45]:


df['datetime'] = pd.to_datetime(df['time_stamp'])
df = df.set_index('datetime')
df.drop(['time_stamp'], axis=1, inplace=True)
df.head()


# In[46]:


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

# In[47]:


df.shape


# In[48]:


df.columns = ['precipMM']


# In[49]:


#get_ipython().run_line_magic('matplotlib', 'inline')
df.plot()


# In[62]:


plt.figure(figsize=(25, 20))
result = seasonal_decompose(df, model='additve')
fig = result.plot()


# In[66]:


# install neccesary packages
#get_ipython().system('pip3 install pmdarima')


# In[67]:



# In[64]:


stepwise_model = auto_arima(df, start_p=1, start_q=1,
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


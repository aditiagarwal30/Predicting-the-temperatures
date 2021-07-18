#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement 1
# ### Perform an exploratory data analysis of your data to check if there is any global warming ?

# In[1]:


import pandas as pd   ## for data manipulaton
import numpy as np     ## for visualization of data
import plotly.express as px
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


global_temp= pd.read_csv('C:\\Users\\devesh\\Desktop\\Git Repositories\\Predicting-the-temperatures\\Dataset\\GlobalTemperatures.csv')


# In[3]:


global_temp.head()


# In[4]:


## how we will fetch year from the given data
global_temp['dt'][0].split('-')[0]


# In[5]:


def fetch_year(date):
    return date.split('-')[0]


# In[6]:


global_temp['years']=global_temp['dt'].apply(fetch_year)


# In[7]:


global_temp.head()


# In[8]:


data= global_temp.groupby('years').agg({'LandAverageTemperature':'mean','LandAverageTemperatureUncertainty':'mean'}).reset_index()


# In[9]:


data


# In[10]:


data['Uncertainity_top']= data['LandAverageTemperature'] + data['LandAverageTemperatureUncertainty']
data['Uncertainity_bottom']= data['LandAverageTemperature'] - data['LandAverageTemperatureUncertainty']


# In[11]:


data.head()


# In[12]:


data.columns


# In[13]:


figure= px.line(data,x='years',y=['LandAverageTemperature',
       'Uncertainity_top', 'Uncertainity_bottom'],title='Average Land Temperature in World')
figure.show()


# ## Problem Statement 2
# ### Explore Average temperature in each season.

# In[14]:


global_temp.dtypes      #datatypes of the columns


# In[15]:


##changing the datatype of column 'dt' in the given dataframe
global_temp['dt']=pd.to_datetime(global_temp['dt']) 


# In[16]:


global_temp.dtypes


# In[17]:


global_temp['month']= global_temp['dt'].dt.month


# In[18]:


global_temp.head()


# In[19]:


def get_season(month):
    if month>=3 and month<=4:
        return 'spring'
    elif month>=5 and month<=9:
        return 'summer'
    elif month>=10 and month<=11:
        return 'autumn'
    else:
        return 'winter'


# In[20]:


global_temp['season']= global_temp['month'].apply(get_season)


# In[21]:


global_temp.head()


# In[22]:


## fetching the unique value in a certain column

years= global_temp['years'].unique()


# In[23]:


years


# In[24]:


spring_temp=[]
summer_temp=[]
autumn_temp=[]
winter_temp=[]


# In[25]:


for year in years:
    current_df=global_temp[global_temp['years']==year]
    spring_temp.append(current_df[current_df['season']=='spring']['LandAverageTemperature'].mean())
    summer_temp.append(current_df[current_df['season']=='summer']['LandAverageTemperature'].mean())
    autumn_temp.append(current_df[current_df['season']=='autumn']['LandAverageTemperature'].mean())
    winter_temp.append(current_df[current_df['season']=='winter']['LandAverageTemperature'].mean())


# In[26]:


season = pd.DataFrame()


# In[27]:


season['year']=years
season['spring_temps']= spring_temp
season['summer_temp']= summer_temp
season['autumn_temp']= autumn_temp
season['winter_temp']= winter_temp


# In[28]:


season.head()


# In[29]:


season.columns


# In[30]:


fig= px.line(season, x='year', y=['spring_temps', 'summer_temp', 'autumn_temp', 'winter_temp'], title='Average Temperature in each Season')
fig.show()


# ## Problem Statement 3
# ### Perform Pre-Processing on your data and make it ready for Time Series Analysis

# In[31]:


cities= pd.read_csv("C:\\Users\\devesh\\Desktop\\Git Repositories\\Predicting-the-temperatures\\Dataset\\GlobalLandTemperaturesByCity.csv")


# In[32]:


cities.tail()


# In[33]:


cities.shape


# In[34]:


usa= cities[cities['Country']=='United States']


# In[35]:


usa.shape


# In[36]:


df= ['New York','Los Angeles','San Francisco']


# In[37]:


df2= usa[usa['City'].isin(df)]


# In[38]:


df2.head()


# In[39]:


df2= df2[['dt','AverageTemperature']]


# In[40]:


df2.head()


# In[41]:


#updating column name


# In[42]:


df2.columns=['Date','AvgTemp']


# In[43]:


df2.head()


# In[44]:


df2.dtypes


# In[45]:


df2['Date']=pd.to_datetime(df2['Date'])


# In[46]:


df2.head()


# In[47]:


df2.isna().sum()     ## will calculate the missing values in this 


# In[48]:


df2.dropna(inplace=True)


# In[49]:


df2.shape


# In[50]:


df2.set_index('Date',inplace=True)


# In[51]:


df2.head()


# ## Problem Statement 4
# ###  How to check whether data is stationary or not ?
# wrt to Time Series use case, it is must that our data follows a stationary nature.
# If our mean and standard deviation of our data gives a pattern which is constant throughout our data then the data is said to be stationary.

# In[52]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


plt.figure(figsize=(20,10))
sns.lineplot(x=df2.index, y=df2['AvgTemp'])


# In[54]:


from statsmodels.tsa.stattools import adfuller


# In[55]:


adfuller(df2['AvgTemp'])


# In[56]:


## we will write a function to fetch individual
## values returned by ADFULLER function


# In[57]:


def Adfuller_Test(AvgTemp):
    result= adfuller(AvgTemp)
    labels= ['ADF Test stats','p-value','#lags used','no. of observations used']
    
    for value,label in zip(result,labels):
        print('{} : {}'.format(label,value))
        
    if result[1]<0.05:
        print('Strong Evidence against the null hypothesis, hence we will reject the hypothesis, data is -----STATIONARY-----')
    else :
        print('Weak Evidence against the null hypothesis, hence we fail will reject the hypothesis, data is -----NOT STATIONARY-----')


# In[58]:


Adfuller_Test(df2['AvgTemp'])


# ## Problem Statement 5
# ### How to make your Data Stationary ?
# 

# In[59]:


## making a copy of df2 so that the data isnt lost upon manipulations


# In[60]:


DF= df2.copy()


# In[61]:


DF.head()


# In[62]:


DF['first_diff_temp']= DF['AvgTemp']-DF['AvgTemp'].shift(12)


# In[63]:


DF.head()


# In[64]:


Adfuller_Test(DF['first_diff_temp'].dropna())


# In[65]:


## task is completed


# In[66]:


DF[['first_diff_temp']].plot(figsize=(16,10))


# ## Problem Statement 6
# ### Examine seasonality in a Data
# 

# In[67]:


df2['month']=df2.index.month


# In[68]:


df2['year']= df2.index.year


# In[69]:


df2.head()


# In[70]:


pivot= df2.pivot_table(values='AvgTemp',index='month',columns='year')


# In[71]:


pivot


# In[72]:


pivot.plot(figsize=(20,6))
plt.legend().remove()
plt.xlabel('Months')
plt.ylabel('Temperatures')


# In[73]:


## to make the graph cleaner


# In[74]:


monthly_seasonality= pivot.mean(axis=1)


# In[75]:


monthly_seasonality.plot(figsize=(20,6))


# ## Problem Statement 7
# ### Build Times Series Model using Moving Average method.

# In[76]:


DF.head()


# In[77]:


DF= DF[['first_diff_temp']]


# In[78]:


DF.head()


# In[79]:


DF.dropna(inplace=True)


# In[80]:


DF.head()


# In[81]:


DF['first_diff_temp'].rolling(window=5).mean()


# In[82]:


value= pd.DataFrame(DF['first_diff_temp'])


# In[83]:


temp_DF= pd.concat([value,DF['first_diff_temp'].rolling(window=5).mean()],axis=1)


# In[84]:


temp_DF.columns= ['actual_temp','forecast_temp']


# In[85]:


temp_DF.head()


# ## Problem Statement 7
# ### Evalute the Moving Average Model

# In[86]:


from sklearn.metrics import mean_squared_error


# In[87]:


np.sqrt(mean_squared_error(temp_DF['forecast_temp'][4:],temp_DF['actual_temp'][4:]))


# In[88]:


## if there is some outlier in the data then we will use the ARIMA model


# ## Problem Statement 8
# ### Evaluate the ARIMA Model

# In[89]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[90]:


plot_acf(DF['first_diff_temp'].dropna())


# In[91]:


plot_pacf(DF['first_diff_temp'].dropna())


# In[92]:


DF.isna().sum()


# In[93]:


DF.shape


# In[94]:


training_data= DF[0:6000]
test_data= DF[6000:]


# In[95]:


from statsmodels.tsa.arima_model import ARIMA


# In[96]:


arima= ARIMA(training_data, order=(2,1,3))


# In[97]:


model= arima.fit()


# In[98]:


model.forecast(steps=10)[0]


# In[99]:


predictions= model.forecast(steps=len(test_data))[0]


# In[100]:


np.sqrt(mean_squared_error(test_data, predictions))


# ## Problem Statement 9
# ### Cross Validation our Time Series Model

# In[101]:


## whichever triplet of (p,d,q) gives the least error will be the best one.


# In[102]:


p_value= range(0,4)
d_value= range(0,4)
q_value= range(0,3)


# In[ ]:


for p in p_value:
    for d in d_value:
        for q in q_value:
            order=(p,d,q)
            train=DF[0:6000]
            test=DF[6000:]
            predictions=[]
            for i in range(len(test)):
                try:
                    arima=ARIMA(train,order)
                    model=arima.fit(disp=0)
                    pred= model.forecast()[0]
                    print(pred)
                    predictions.append(pred)
                    mean_squared_error(test,predictions)
                    print(f'MSE is {error} with order {order}')
                except:  
                    continue




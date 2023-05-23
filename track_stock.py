#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:





# In[2]:


dataset=pd.read_csv('D:/tesla.csv')


# In[3]:


dataset.head()


# In[4]:


dataset['Date'] = pd.to_datetime(dataset.Date)


# In[5]:


dataset.shape


# In[6]:


dataset.columns = ['date','Open', 'High', 'Low', 'Close', 'Volume']
dataset.index.name = "Date"


# In[7]:


dataset['Date'] = pd.to_datetime(dataset.Date)


# In[8]:


dataset.isnull().sum()


# In[9]:


dataset.info()


# In[10]:


dataset.isna().any()


# In[11]:


dataset.describe()


# In[12]:


print(len(dataset))


# In[13]:


dataset['Open'].plot(figsize=(16,6))


# In[14]:


X=dataset[['Open','High','Low','Volume']]
y=dataset['Close']


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random state = 0)


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)


# In[17]:


X_train.shape


# In[18]:


X_test.shape


# In[19]:


regressor.fit(X_train,y_train)


# In[20]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
regressor = LinearRegression()


# In[21]:


regressor.fit(X_train,y_train)


# In[22]:


print(regressor.coef_)


# In[23]:


print(regressor.intercept_)


# In[24]:


predicted=regressor.predict(X_test)


# In[25]:


print(X_test)


# In[26]:


predicted.shape


# In[27]:


dframe=pd.DataFrame(y_test,predicted)


# In[28]:


dfr=pd.DataFrame({'Actual Price':y_test,'Predicted Price':predicted})


# In[29]:


print{dfr}


# In[30]:


print(dfr)


# In[31]:


dfr.head(25)


# In[32]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[33]:


regressor.score(X_test,y_test)


# In[34]:


import math


# In[35]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predicted))


# In[36]:


print('Mean Squared Error:', metrics.mean_squared_error(y_test, predicted))


# In[37]:


print('Root Mean Squared Error:', math.sqrt(metrics.mean_squared_error(y_test, predicted)))


# In[38]:


print('Root Mean Squared Error:', math.sqrt(metrics.mean_squared_error(y_test, predicted)))


# In[39]:


graph.plot(kind='bar')


# In[40]:


graph=dfr.head(20)


# In[41]:


graph.plot(kind='bar')


# In[ ]:





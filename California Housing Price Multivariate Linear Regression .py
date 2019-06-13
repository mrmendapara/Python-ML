#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output


# In[7]:


data = pd.read_csv("G:\DATA SCIENCE\Assignment\For Python\california-housing-prices\housing.csv")


# In[8]:


data


# In[22]:


target_var = data['median_house_value']
feature = data.drop(['median_house_value','ocean_proximity'], axis=1).copy()


# In[23]:


feature = feature/feature.max()
target_var = target_var/target_var.max()


# In[24]:


target_var = np.array(target_var)
target_var = np.reshape(target_var, [1,len(target_var)])
print(target_var.shape)


# In[25]:


feature = np.array(feature)


# In[26]:


print("Feature Shape : ", feature.shape)
print("Target Variable shape : ", target_var.shape)


# In[27]:


def line_multidim(m,x,c):
  return np.dot(m,x.T) + c

def error(m,x,c,y):
  return np.mean((line_multidim(m,x,c) - y)**2)

def derivative_slopes(m,x,c,y):
  return 2*np.mean(np.multiply((line_multidim(m,x,c)-y),x.T),axis = 1)

def derivative_intercept(m,x,c,y):
  return 2*np.mean((line_multidim(m,x,c)-y))

def accuracy_pred(error,y):
  return 100 - (error/np.mean(y**2))*100


# In[28]:


m = np.random.randn(1,8)
c = random.random()

iterations = 2000
lr = 0.01
error_array = []

for i in range(0,iterations):
  m = m - lr*derivative_slopes(m,feature,c,target_var)
  c = c - lr*derivative_intercept(m,feature,c,target_var)
  error_array.append(error(m,feature,c,target_var))
  clear_output(True)
  print("Current Error: ",error(m,feature,c,target_var),"Current Iteration:",i)
  print("Current Accuracy:",accuracy_pred(error(m,feature,c,target_var),target_var))
  
plt.plot(error_array)
plt.show()


# In[ ]:





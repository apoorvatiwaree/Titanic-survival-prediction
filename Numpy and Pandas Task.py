#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
data=pd.read_csv("owid-covid-data.csv")
print(data.head())


# In[3]:


print(data.tail)


# In[4]:


print(data.info())


# In[5]:


data.dtypes


# In[6]:


data.describe()


# In[10]:


""""Data of how many countries is present"""
print(len(pd.unique(data['location'])))


# In[11]:


"""Number of continents"""
print(len(pd.unique(data['continent'])))


# In[13]:


"""How many rows belong to India"""
print(len(data[data['location']=='India']))


# In[14]:


"""Window of dates"""
print(min(data['date'])," to ",max(data['date']))


# In[26]:


"""
Extract Data of only India and make into a new dataframe.
Extract only the total cases column.
Convert total cases column into percentage. Percentage of total cases. Total cases is the number of cases as on the last date of the dataset.
"""

india_data=data[data['location']=='India']
total_cases=data['total_cases']
total=total_cases.tail(1)
print(total_cases.info())


# In[ ]:





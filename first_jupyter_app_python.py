#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the dataset
import pandas as pd
import sys
file = sys.argv[1]
print(file)
dataset = pd.read_csv(file)
print(dataset)


# In[4]:


#Visualizing the Linear Regression results
import matplotlib.pyplot as plt
plt.scatter(dataset[['x']], dataset[['y']], color = 'green')
plt.title('Python:1st Jupyter App')
plt.xlabel('X values from regrex1.csv')
plt.ylabel('Y values from regrex1.csv')
plt.show()


# In[10]:


# Fitting Linear Regression to the Dataset
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])

#Visualizing the Linear Regression results
#import matplotlib.pyplot as plt
plt.scatter(dataset[['x']], dataset[['y']], color = 'teal')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'red')
plt.title('Python:1st Jupyter App [MODELED]')
plt.xlabel('X-values  from regrex1.csv')
plt.ylabel('Y-values from regrex1.csv')
plt.show()


# In[ ]:





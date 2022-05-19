#!/usr/bin/env python
# coding: utf-8

# # Assignment on Linear Regression:
# The following table shows the results of a recently conducted study on the correlation of the
# number of hours spent driving with the risk of developing acute backache. Find the equation of
# the best fit line for this data.
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


#input data
x = np.array([10,9,2,15,10,16,11,16])
y=np.array([95,80,10,50,45,98,38,93])
plt.scatter(x,y)


# In[3]:


def estimate_coefficient(x,y):

    N =np.size(x)
    
    x_mean,y_mean=np.mean(x),np.mean(y);
   
    ss_xy=np.sum(y*x)-N*x_mean*y_mean;
    ss_xx=np.sum(x*x)-N*x_mean*x_mean;
    
    b1=ss_xy/ss_xx;
    b0=y_mean-b1*x_mean;
    
    return (b0,b1);
    


# In[4]:


def plot_regression_line(x,y,b):
   
    plt.scatter(x,y, color="m", marker="o",s=30)
   
    y_pred=b[0]+b[1]*x;
    
    plt.plot(x,y_pred, color="g");
    
    plt.xlabel("X")
    plt.ylabel("Y")
    
    plt.show() 


# In[5]:


b=estimate_coefficient(x,y)
print("Estimated coefficients:\n b0 = {} \n b1 = {}".format(b[0],b[1]))


# In[6]:


plot_regression_line(x,y,b)


# In[7]:


from sklearn.linear_model import LinearRegression


# In[8]:


model=LinearRegression().fit(np.reshape(x,(-1,2)),np.reshape(y,(-1,2)))
model.coef_


# In[9]:


model.intercept_


# In[10]:


model.score(np.reshape(x,(-1,2)),np.reshape(y,(-1,2)))


# In[ ]:





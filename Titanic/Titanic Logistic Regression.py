#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# In[2]:


titanic=pd.read_csv(r"C:\Users\Home\Downloads\0000000000002429_training_titanic_x_y_train.csv",delimiter=",")
titanic.columns


# In[3]:


def f(s):
    if s=="female":
        return 0
    else:
        return 1
       
titanic["Gender"]=titanic.Sex.apply(f)
del titanic["Sex"]


# In[4]:


del titanic["Cabin"]


# In[5]:


titanic.Age.fillna(titanic.Age.median(),inplace=True)


# In[6]:


titanic.drop("Name",inplace=True,axis=1)


# In[7]:


def g(s):
    if s=="C":
        return 0
    elif s=="Q":
        return 1
    elif s=="S":
        return 2
titanic["Embark"]=titanic.Embarked.apply(g)


# In[8]:


del titanic["Ticket"]


# In[9]:


titanic["Survive"]=titanic["Survived"]


# In[10]:


del titanic["Survived"]


# In[11]:


del titanic["Embarked"]


# In[12]:


titanic.dropna(inplace=True)
titanic.shape


# In[13]:


titanic=titanic.values


# In[14]:


x=titanic[:,0:7]
y=titanic[:,7]
y=y.astype(int)


# In[15]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(x,y)


# In[16]:


xtest=pd.read_csv(r"C:\Users\Home\Downloads\0000000000002429_test_titanic_x_test.csv",delimiter=",")
xtest.Age.fillna(xtest.Age.median(),inplace=True)
xtest["Gender"]=xtest.Sex.apply(f)
xtest["Embark"]=xtest.Embarked.apply(g)
del xtest["Sex"]
del xtest["Embarked"]
xtest=xtest.dropna()
ypred=clf.predict(xtest)


# In[17]:


np.savetxt(r"C:\Users\Home\Desktop\AssignmentLog.csv",ypred,delimiter=",")


#!/usr/bin/env python
# coding: utf-8

# # fetching dataset

# In[48]:


from sklearn.datasets import fetch_openml


# In[2]:


import matplotlib


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import numpy as np


# In[5]:


from sklearn.linear_model import LogisticRegression


# In[6]:


from sklearn.model_selection import cross_val_score


# In[39]:


import pandas as pd


# In[40]:


mnist = fetch_openml('mnist_784')


# In[41]:


x, y = mnist['data'], mnist['target']


# In[42]:


x.shape


# In[11]:


y.shape


# In[12]:


some_digit = x.to_numpy()[36001]
some_digit_image = some_digit.reshape(28, 28)  # let's reshape to plot it


# In[13]:


plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
           interpolation='nearest')
plt.axis("off")
plt.show()


# In[14]:


y[36001]


# In[15]:


y[36000]


# In[16]:


x_train, x_test = x[0:6000], x[6000:7000]


# In[17]:


y_train, y_test = y[0:6000], y[6000:7000]


# In[18]:


shuffle_index = np.random.permutation(60000)


# # Creating a 2-detector

# In[19]:


y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)


# In[20]:


y_test_2


# # Train a logistic regression classifier

# In[ ]:


clf = LogisticRegression(tol=0.1, solver='1bfgs')


# In[ ]:


clf.fit(x_train, y_train_2)


# In[23]:


example = clf.predict([some_digit])


# In[24]:


print(example)


# # Cross Validation

# In[25]:


a = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")


# In[47]:


print(a.mean())


# In[ ]:





# In[ ]:





# In[ ]:





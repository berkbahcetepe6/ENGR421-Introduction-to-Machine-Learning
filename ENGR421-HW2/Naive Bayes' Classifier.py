#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import scipy.linalg as linalg


# In[2]:


data_set_labels = np.genfromtxt("hw02_labels.csv")


# In[3]:


data_set_images = np.genfromtxt("hw02_images.csv",delimiter=',')


# In[ ]:





# In[4]:


#groups the data and their labels
x_train = data_set_images[:30000,:]
y_train = data_set_labels[:30000]
x_test = data_set_images[30000:,:]
y_test = data_set_labels[30000:]


# In[5]:


K = np.max(y_train).astype(int)
N = x_train.shape[0]


# In[ ]:





# In[6]:


#calculates the sample means
sample_means= np.array([])
for i in range(K):
    for j in range(x_train.shape[1]):
        mean=np.mean(x_train[y_train==(i+1)][:,j])
        sample_means=np.append(sample_means,mean)
        
sample_means=sample_means.reshape(5,784)


# In[7]:


print(sample_means)


# In[8]:


#calculates the sample deviations
sample_deviations= np.array([])
for i in range(K):
    for j in range(x_train.shape[1]):
        std=np.sqrt(np.mean((x_train[y_train==(i+1)][:,j]-sample_means[i,j])**2))
        sample_deviations=np.append(sample_deviations,std)
        
sample_deviations=sample_deviations.reshape(5,784)


# In[9]:


print(sample_deviations)


# In[10]:


#calculates the class priors
class_priors = [np.mean(y_train==(c+1)) for c in range(K)]


# In[11]:


print(class_priors)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


#calculates the gscore functions for each class
def g_calc(x,mu,std,prior):
    scores = np.array([])
    for i in range(x.shape[0]):
        for c in range(K):
            score = np.array(np.sum(-0.5*np.log(2*math.pi*mu[c]**2)-0.5*(x[i]-mu[c])**2/std[c]**2 + np.log(prior[c])))
            scores = np.append(scores,score)
    return scores.reshape(x.shape[0],K)


# In[13]:


y_train_pred = g_calc(x_train,sample_means,sample_deviations,class_priors)


# In[14]:


#confusion matrix for the training part
y_train_predicted = np.argmax(y_train_pred, axis = 1) + 1
confusion_matrix = pd.crosstab(y_train_predicted, y_train, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)


# In[15]:


y_test_pred = g_calc(x_test,sample_means,sample_deviations,class_priors)


# In[16]:


#confusion matrix for the test part
y_test_predicted = np.argmax(y_test_pred, axis = 1) + 1
confusion_matrix1 = pd.crosstab(y_test_predicted, y_test, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





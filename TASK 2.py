#!/usr/bin/env python
# coding: utf-8

# # GRIP- THE SPARKS FOUNDATION
# 

# # TASK 2 - Prediction using Unsupervised ML

# # From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
from sklearn.cluster import KMeans


# In[7]:


iris = datasets.load_iris()
print("Dataset loaded successfully")


# In[8]:


#Creating data frame 
Data = pd.DataFrame(iris.data, columns = iris.feature_names)

#Top values of Dataset
Data.head()


# In[9]:


#Bottom Values of Dataset
Data.tail()


# In[10]:


Data.shape


# In[11]:


Data.info()


# In[12]:


Data.describe()


# In[14]:


sns.heatmap(Data.corr(), annot = True, linecolor='red')


# In[15]:


Data.hist()
plt.show()


# In[16]:


# Settin the data
x=Data.iloc[:,0:3].values

css=[]

# Finding inertia on various k values
for i in range(1,8):
    kmeans=KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 100, n_init = 10, random_state = 0).fit(x)
    css.append(kmeans.inertia_)
    
plt.plot(range(1, 8), css, 'bx-', color='red')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('CSS') 
plt.show()


# In[18]:


kmeans = KMeans(n_clusters=3,init = 'k-means++', max_iter = 100, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(x)


# In[19]:


kmeans.cluster_centers_


# In[20]:


# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'black', label = 'Centroids')

plt.legend()


# # Optimum number of clusters are represented in the above plot.

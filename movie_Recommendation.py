#!/usr/bin/env python
# coding: utf-8
from IPython import get_ipython


import tensorflow as tf
import tensorflow_hub as hub


# In[7]:


import os 
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


# In[8]:


model_url='https://tfhub.dev/google/universal-sentence-encoder/4'


# In[11]:


model= hub.load(model_url)
print("MODEL Loaded")


# In[75]:


def bd(texts):
    return model(texts)


# In[76]:


df = pd.read_csv("Top_10000_Movies.csv", engine="python")
df.head()


# In[77]:


df= df[["original_title","overview"]]


# In[78]:


df= df.dropna()
df= df.reset_index()



# In[79]:


titles= list(df['overview'])


# In[80]:


embeddings= bd(titles)
print("shape", embeddings.shape)


# In[81]:


pca= PCA(n_components=2)
emb_2d= pca.fit_transform(embeddings)


# In[82]:


plt.figure(figsize=(11,6))
plt.title('Embeddings')
plt.scatter(emb_2d[:,0],emb_2d[:,1])
plt.show()


# In[83]:


nn = NearestNeighbors(n_neighbors=10)
nn.fit(embeddings)


# In[84]:


def recommend(text):
    emb=bd([text])
    neighbors= nn.kneighbors(emb,return_distance=False)[0]
    return df['original_title'].iloc[neighbors].tolist()


# In[85]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # 假设
# 
# 全眼的像差，是最终对视网膜产生影响的原因。那么使用佩戴OK镜之前、或者佩戴短期后的像差数据，是否可能预测出远期的眼轴长或者是屈光状态呢？
# 
# ## 已知的缺陷
# 
# 全眼像差受到多方面的影响：
# 
# * 瞳孔大小；
# * 调节状态；
# * 测量时间，OK镜佩戴后，白天的角膜形态是否会逐渐变化，导致像差随着时间改变。
# 
# 

# # 数据
# 
# 来自于 One-year effect of wearing orthokeratology lenses on the visual quality of juvenile myopia: a retrospective study 
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6545229/
# 

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data_file=os.path.join('data',"peerj-07-6998-s001.xlsx")


# In[7]:


df=pd.read_excel(data_file)


# In[9]:


df.head(10)


# 数据清洗：
# 
# 

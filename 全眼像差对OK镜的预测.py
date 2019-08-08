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
# 来自于[Predictive factors associated with axial length growth and myopia progression in orthokeratology ](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6561598/ )
# 
# 该文献带有excel数据，共有7个sheet，分别是：
# 
# * age, sex, visual acuity：年龄，性别，视力，其中视力用LogMAR，包含了未矫正和最佳矫正视力。
# * AXL：眼轴长，用IOL master测量了中央，鼻侧30度，颞侧30度
#   >AXL measurement with IOLMaster (Carl Zeiss, Jena, Germany) in central, N30, and T30 gazes
# * CR：散瞳验光，用WAM-5500测量了中央，鼻侧30度，颞侧30度
#   >cycloplegic refraction; autorefraction (WAM-5500; Shigiya Machinery Works Ltd., Hiroshima, Japan) in central, 30° nasal (N30), and 30° temporal (T30) gazes under cycloplegia
# * MR：
#   >manifested refraction
# * specular microscopy：不知为何，测量了角膜内皮细胞计数。
#   >evaluation of the corneal endothelium via noncontact specular microscopy (SP-8000; Konan Medical, Nishinomiya, Japan). 
# * aberrometer：像差，给了高阶的Zernike系数。
#   >wavefront assessment for a 6-mm pupil using a WASCA aberrometer (Carl Zeiss, Jena, Germany) following pupil dilation using a mixture of 0.5% phenylephrine and 0.5% tropicamide (Mydrin-P; Santen Pharmaceutical, Osaka, Japan)
# * pentacam：角膜地形图。
#   很遗憾，这里面不是raw data，只有Pre和12mo的K1, K2
# * orbscan II：角膜地形图。
#   也不是角膜地形图的原始数据，但除了Kmin, Kmax,还有Central corneal thickness, 3-mm-zone irregularity, 5-mm-zone irregularity

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 数据清洗
# 

# In[21]:


data_file=os.path.join('data',"pone.0218140.s001.xlsx")
AXL=pd.read_excel(data_file,sheet_name="AXL")
CR=pd.read_excel(data_file,sheet_name="CR ")
aberrometer=pd.read_excel(data_file,sheet_name="aberrometer")
cornea=pd.read_excel(data_file,sheet_name="orbscan II")


# In[19]:





# 数据清洗：
# 
# 

# In[5]:


df.columns


# In[6]:


# target_df=df["Spherical Diopter"].where(df["Right Eye"]=="12 months").dropna()
# sns.distplot(target_df)


# In[ ]:





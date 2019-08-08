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

# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce


import os

get_ipython().run_line_magic('matplotlib', 'inline')


# ## 数据清洗
# 
# 读取数据后，将数据分成两部分，（务必要注意是否有数据泄露）
# 
# * X：从这些数据可能推导出结果，我估计会有术前的数据，一部分术后的数据。
#     * patient_info中：``` ['Age','Sex (male = 1, female = 2']```
#     * AXL：```['Pre C AXL', 'Pre N AXL', 'Pre T AXL'] ```
#     * CR：
#       ```python
#          ['Pre AR C Sph', 'Pre AR C cyl',\
#           'Pre AR N Sph', 'Pre AR N cyl',\
#           'Pre AR T Sph', 'Pre AR T cyl']  
#        ```
#     * aberrometer:
#         * Pre和12mo （犹豫，不知道是否有数据泄露）
#     * cornea：
#         * Pre和12mo
# * Y：
#     * AXL：12mo的C，N，T，以及delta，其中delta 12mo C AXL是最重要的数据。

# In[89]:


data_file=os.path.join('data',"pone.0218140.s001.xlsx")

patient_info=pd.read_excel(data_file,sheet_name="age, sex, visual acuity")
AXL=pd.read_excel(data_file,sheet_name="AXL")
CR=pd.read_excel(data_file,sheet_name="CR ")

# 以下两个sheet中，顶部有Pre，12mo一行，
# 略去，使得每一行与其他表格中的行位置相等。
aberrometer=pd.read_excel(data_file,sheet_name="aberrometer",header=1) 
cornea=pd.read_excel(data_file,sheet_name="orbscan II",header=1) 
data_frames=[patient_info,AXL,CR,aberrometer,cornea]


# 并不是所有的人都测量了所有的参数，所以将Patient ID和眼别整合到一起，形成一个新的eyeID。

# In[91]:


for d in data_frames:
    d["Patient"].fillna(method='ffill',inplace = True)
    d["eyeID"]=d["Patient"]+" "+d['OD1, OS2'].map(str)


# In[95]:


df = reduce(lambda left,right: pd.merge(left,right,on='eyeID'), data_frames)


# In[96]:


df


# In[ ]:





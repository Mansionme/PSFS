## -*- coding: utf-8 -*-
"""
This code is created to distinguish university students having financial difficulties 
 (EDA (EXPLORATORY DATA ANALYSIS) PART)
Created on Tue Mar  2 11:52:27 2021


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

data = pd.read_excel(r'data\数据清洗.xlsx')
y = data['是否在校申请贫困']
# x = data.drop(['Unnamed: 0','是否在校申请贫困','突发事件','家庭负债','温饱问题','网购','失地农民','失业','城镇农民工','谈恋爱','护肤或者化妆品','家庭受非义务教育人数'],axis=1)
x = data.drop(['Unnamed: 0','是否在校申请贫困'],axis=1)
# 热力图
x_zscore = (x - x.mean())/x.std()
data_new = x
data_new['是否在校申请贫困'] = y
data_new_zscore = (data_new - data_new.mean())/data_new.std()
corr = data_new_zscore.corr('spearman')
# corr = data_new_zscore.corr('pearson')
plt.figure(figsize=(10,10))
fig = sns.heatmap(corr,cmap='coolwarm',linewidths=.55)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.title(r"（部分）特征与标签的$Spearman$相关系数矩阵",fontsize=20)
plt.title(r"特征与标签的$Spearman$相关系数矩阵",fontsize=20)
plt.subplots_adjust(left=0.25, wspace=0.25, hspace=0.25,bottom=0.38, top=0.95)
# plt.savefig('heatmap2.pdf',dpi=1000)
plt.savefig('heatmap.pdf',dpi=1000)

# PCA  在本数据集上效果不好
pca = PCA(n_components=8)
X_r = pca.fit(x_zscore).transform(x_zscore)
print(np.cumsum(pca.explained_variance_ratio_))

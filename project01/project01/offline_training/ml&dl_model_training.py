## -*- coding: utf-8 -*-
"""
This code is created to distinguish university students having financial difficulties  
 (MACHINE LEARNING & DEEP LEARNING MODELING PART)
Created on Tue Mar  2 07:43:59 2021

@author: Zjx2019
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import xgboost
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
# matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
plt.style.use('ggplot')


# dl绘图函数
def draw_dl_figures(history_dict):
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1,len(loss_values)+1)
    plt.plot(epochs,loss_values,'bo',label='Training loss')
    plt.plot(epochs,val_loss_values,'b',label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.clf()
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    plt.plot(epochs,acc_values,'bo',label='Training acc')
    plt.plot(epochs,val_acc_values,'b',label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()

source_data = pd.read_excel(r'data\数据清洗.xlsx')


'''
机器学习建模
'''

y = source_data['是否在校申请贫困']
# 如果样本量过大，删除一些与y的相关系数较小的变量也可以，如下所示（这是根据相关性矩阵算出来的）
# x = source_data.drop(['Unnamed: 0','是否在校申请贫困','突发事件','家庭负债','温饱问题','网购','失地农民','失业','城镇农民工','谈恋爱','护肤或者化妆品','家庭受非义务教育人数'],axis=1)
x = source_data.drop(['Unnamed: 0','是否在校申请贫困'],axis=1)
# print(x.shape)

# poly = PolynomialFeatures(degree=2)
# x = poly.fit_transform(x)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.25,random_state=0)
xtrain_scaled, xtest_scaled, ytrain_scaled, ytest_scaled = train_test_split(x_scaled,y,test_size=0.25,random_state=0)

# 随机森林
RFmodel = RandomForestClassifier(n_estimators= 42, max_depth=7, min_samples_split=5,
                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10)
RFmodel.fit(xtrain,ytrain)
RFscore = RFmodel.score(xtest,ytest)
print(RFscore)
# # joblib.dump(RFmodel,'saved_model/RFmodel.pkl')

# # 绘图
# plt.figure(figsize=(10,5.5))
# fig = plt.barh(y = x.columns, width = RFmodel.feature_importances_, color='#FA8072',ec='black',lw=.45)
# fig = plt.barh(y = x.columns[6], width = RFmodel.feature_importances_[6], color='#E61A1A',ec='black',lw=.45,alpha=.9)
# fig = plt.barh(y = x.columns[14], width = RFmodel.feature_importances_[14], color='#E61A1A',ec='black',lw=.45,alpha=.9)
# fig = plt.barh(y = x.columns[8], width = RFmodel.feature_importances_[8], color='#E61A1A',ec='black',lw=.45,alpha=.9)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title('随机森林特征重要性',fontsize=22)
# plt.subplots_adjust(left=0.25, wspace=0.25, hspace=0.25,
#                     bottom=0.13, top=0.91)
# # plt.savefig('随机森林特征重要性.pdf',dpi=1000)
# plt.show()

# plt.figure(figsize=(10,5.5))
# order = np.argsort(RFmodel.feature_importances_)
# fig = plt.barh(y = x.columns[order], width = RFmodel.feature_importances_[order], color='#FA8072',ec='black',lw=.45)
# fig = plt.barh(y = x.columns[order][-1], width = RFmodel.feature_importances_[order][-1], color='#E61A1A',ec='black',lw=.45,alpha=.9)
# fig = plt.barh(y = x.columns[order][-2], width = RFmodel.feature_importances_[order[-2]], color='#E61A1A',ec='black',lw=.45,alpha=.9)
# fig = plt.barh(y = x.columns[order][-3], width = RFmodel.feature_importances_[order[-3]], color='#E61A1A',ec='black',lw=.45,alpha=.9)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.title('随机森林特征重要性（排序）',fontsize=22)
# plt.subplots_adjust(left=0.25, wspace=0.25, hspace=0.25,
#                     bottom=0.13, top=0.91)
# # plt.savefig('随机森林特征重要性(排序).pdf',dpi=1000)
# plt.show()

# fig = plt.figure(figsize=(10,5.5))
# cumsum = np.cumsum(RFmodel.feature_importances_[order][::-1])
# plt.scatter(x=range(20),y=cumsum,marker='o',s=45,c='white',edgecolor='black',linewidths=1,alpha=1)
# plt.plot(cumsum,linewidth=2.8,label='特征贡献率累和',marker='o',markersize=5)
# plt.plot([-1,21],[0.8,0.8],linewidth=1.5,linestyle='dashed')
# loc = range(20)
# plt.xticks(loc, list(x.columns.values[order][::-1]),rotation=90,fontsize=15)
# plt.title('特征帕累托分布',fontsize=22)
# plt.xlim([-1,21])
# plt.ylim([0.33,1.05])
# plt.legend(fontsize=14,fancybox=True,framealpha=.9,shadow='True')
# plt.subplots_adjust(left=0.25, wspace=0.25, hspace=0.25,
#                     bottom=0.53, top=0.91)
# ax=plt.gca()
# y_major_locator=MultipleLocator(0.1)
# ax.yaxis.set_major_locator(y_major_locator)
# # plt.savefig('特征帕累托分布.pdf',dpi=1000)
# plt.show()

# 线性核SVC
C = 1.3
LinearSVCmodel = SVC(kernel='linear', C=C, gamma='auto',probability=True)
LinearSVCmodel.fit(xtrain_scaled,ytrain_scaled)
LinearSVCscore = LinearSVCmodel.score(xtest_scaled,ytest_scaled)
sp_v = LinearSVCmodel.support_vectors_
print(LinearSVCscore)
# # joblib.dump(LinearSVCmodel,'saved_model/LinearSVCmodel.pkl') # save model

# # 存图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xtrain_scaled[:120,6],xtrain_scaled[:120,7],xtrain_scaled[:120,8],c=ytrain_scaled[:120],s=130,label= 'P&N Samples',edgecolors='white',cmap='coolwarm_r')
# ax.scatter(sp_v[:,6],sp_v[:,14],sp_v[:,8],c='r',label='Support Vectors',s=80)
# ax.view_init(elev=0, azim=0)  # 观察角
# plt.title('Support Vectors Modeled by \n Support Vector Machines Method(2)',fontsize=14)
# plt.legend(loc=3,framealpha=.8,shadow='True',fancybox=True)
# plt.savefig('svm2.pdf',dpi=1000)
# plt.show()

# XGB
XGBmodel = XGBClassifier(learning_rate=0.1,n_estimators=70,max_depth=5,min_child_weight=6,gamma=0,subsample=0.8,  
                          colsample_bytree=0.8,objective= 'binary:logistic',scale_pos_weight=0.8,seed=27, use_label_encoder=False,n_jobs=-1)
XGBmodel.fit(xtrain, ytrain)
y_pre = XGBmodel.predict(xtest)
y_pre = [round(value) for value in y_pre]
XGBscore = accuracy_score(ytest, y_pre)
print(XGBscore)
# # joblib.dump(XGBmodel,'saved_model/XGBmodel.pkl')  # 保存模型

# DeepLearning
DLmodel = keras.models.Sequential([keras.layers.Dense(units=100,activation='selu'),
                                  keras.layers.Dense(100,activation='selu'),
                                  keras.layers.Dense(100,activation='selu'),
                                  keras.layers.Dense(1,activation='sigmoid')])
DLmodel.compile(optimizer = tf.optimizers.RMSprop(lr=0.0002),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

history = DLmodel.fit(xtrain_scaled,ytrain_scaled,epochs=12,validation_data=(xtest_scaled,ytest_scaled))

# dl绘图
history_dict = history.history
keys = history_dict.keys()
# print(keys)
draw_dl_figures(history_dict = history_dict)

DLscores = sum(history_dict['val_accuracy'])/len(history_dict['val_accuracy'])
print(DLscores)  # 验证集得分
# # tf.saved_model.save(DLmodel, "saved_model/DLmodel")  # 保存模型
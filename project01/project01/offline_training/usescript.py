## -*- coding: utf-8 -*-
"""
This code is created to distinguish university students having financial difficulties 
 (USING MODEL PART)
Created on Tue Mar  2 11:52:27 2021


"""


from sklearn.externals import joblib
import tensorflow as tf
import pandas as pd
import numpy as np

# load model
LinearSVCmodel = joblib.load(r'saved_model\LinearSVCmodel.pkl')
RFmodel = joblib.load(r'saved_model\RFmodel.pkl')
XGBmodel = joblib.load(r'saved_model\XGBmodel.pkl')
DLmodel = tf.keras.models.load_model(r'saved_model\DLmodel')

# load sample
# usescript仅提供接口，提供数据后能判定标签，下以问卷清洗后的数据为例
data = pd.read_excel(r'data\数据清洗.xlsx')
sample = data.loc[0]
# x = sample.drop(['Unnamed: 0','是否在校申请贫困'],axis=1)
x = sample.drop(['Unnamed: 0','是否在校申请贫困']).values
x = np.array(x).reshape(-1, x.shape[0])

# classify the sample(make a judgement)
lambda1 = lambda2 = lambda3 = lambda4 = 0.25
y_pre =   (lambda1 * LinearSVCmodel.predict_proba(x)[0][0]+ lambda2 * RFmodel.predict_proba(x)[0][0]+ 
         lambda3 * XGBmodel.predict_proba(x)[0][0] + lambda4 * DLmodel.predict(x)[0][0])
if (y_pre > .5):
    print('This is a negative sample with %.2f percentage probability' %y_pre)
else:
    print('This is a positive sample with %.2f percentage probability' %(1 - y_pre))

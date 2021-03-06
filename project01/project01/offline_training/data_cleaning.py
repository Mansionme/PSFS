# -*- coding: utf-8 -*-
"""
This code is created to distinguish university students having financial difficulties 
 (DATA CLEANING PART)
Created on Mon Mar  1 08:32:27 2021

@author: Zjx2019
"""


import pandas as pd
import numpy as np

source_data = pd.read_excel(r'data\108397328_0_大学生消费情况调查问卷.xlsx')
source_data = pd.DataFrame(source_data)
source_data.drop(columns=['序号', '提交答卷时间', '所用时间', '来源', '来源详情', '来自IP', '1、您的性别', '2、年级','4、家庭所在地'],inplace=True)
# print(source_data,source_data.columns)
# print(source_data.describe())
source_data.columns = ['生活费开销项目','家庭受非义务教育人数','父母工作类型','是否重大疾病','贫困地区','劳动力人口','突发事件','家庭负债','人均月收入','是否在校申请贫困','生活费','基础消费']
(row, col) = source_data.shape
# print((row, col))


# 数据清洗
source_data['生活费开销项目'] = source_data['生活费开销项目'].str.split('┋')
online_shopping = np.zeros(shape=row,dtype=np.int8)
food = np.zeros(shape=row,dtype=np.int8)
entertainment = np.zeros(shape=row,dtype=np.int8)
dating = np.zeros(shape=row,dtype=np.int8)
cosmetics = np.zeros(shape=row,dtype=np.int8)

cost_items = {'网购':online_shopping,'温饱问题':food,'娱乐':entertainment,'谈恋爱':dating,'护肤或者化妆品':cosmetics}
# for item in cost_items:
#     source_data[item] = pd.Series(data = np.zeros(shape=row,dtype=np.int8))

for each in range(row):
    for i in source_data.loc[each]['生活费开销项目']:
        cost_items[i][each] = 1
        # print(source_data.loc[each][i])

# print(online_shopping)
for item in cost_items:
    source_data[item] = pd.Series(data = cost_items[item])

# print(source_data['生活费开销项目'].head())
# print(source_data['网购'].head())
# print(source_data['温饱问题'].head())
source_data.drop(columns = ['生活费开销项目'],inplace=True)

# print(pd.unique(source_data['家庭受非义务教育人数']))
source_data['家庭受非义务教育人数'] = source_data['家庭受非义务教育人数'].map({'1人':1,'2人':2,'3+':3})

source_data['父母工作类型'] = source_data['父母工作类型'].str.split('┋')
farmers = np.zeros(shape=row,dtype=np.int8)
farmers_withoutfarmland = np.zeros(shape=row,dtype=np.int8)
unemployees = np.zeros(shape=row,dtype=np.int8)
merchants = np.zeros(shape=row,dtype=np.int8)
farmer_workers = np.zeros(shape=row,dtype=np.int8)
others = np.zeros(shape=row,dtype=np.int8)
job_items = {'务农':farmers,'失地农民':farmers_withoutfarmland,'失业':unemployees,'个体工商户':merchants,'城镇农民工':farmer_workers,'其它':others}
# for item in job_items:
#     source_data[item] = pd.Series(data = np.zeros(shape=row,dtype=np.int32))
# source_data['其它'] = pd.Series(data = np.zeros(shape=row,dtype=np.int32))

for each in range(row):
    for i in source_data.loc[each]['父母工作类型']:
        if i in job_items:
            job_items[i][each] = 1
        else:
            others[each] = 1
            
for item in job_items:
    source_data[item] = pd.Series(data = job_items[item])
source_data.drop(columns = ['父母工作类型'],inplace=True)

# print(pd.unique(source_data['是否重大疾病']))
source_data['是否重大疾病'] = pd.get_dummies(source_data['是否重大疾病'],prefix='是否重大疾病')
source_data['是否重大疾病'] = abs(source_data['是否重大疾病'] -1)%254
# print(source_data['是否重大疾病'])

# print(pd.unique(source_data['贫困地区']))
source_data['贫困地区'] = pd.get_dummies(source_data['贫困地区'],prefix='贫困地区')
source_data['贫困地区'] = abs(source_data['贫困地区'] -1)%254
# print(source_data['贫困地区'])

# print(pd.unique(source_data['劳动力人口']))
source_data['劳动力人口'] = source_data['劳动力人口'].map({'1':1,'2':2,'3':3,'4+':4})
# print(source_data['劳动力人口'])

source_data['突发事件'] = pd.get_dummies(source_data['突发事件'],prefix='突发事件')
# print(source_data['突发事件'])

# print(pd.unique(source_data['家庭负债']))
source_data['家庭负债'] = source_data['家庭负债'].map({'0':0,'0-5':2.5,'5-10':7.5,'10+':15})

# print(pd.unique(source_data['人均月收入']))
source_data['人均月收入'] = source_data['人均月收入'].map({'6000+':8500,'3000-4000':3500,'5000-6000':5500,'2000-3000':2500,'4000-5000':4500,'1200-2000':1600,'600以下':300,'600-1200':900})

source_data['是否在校申请贫困'] = pd.get_dummies(source_data['是否在校申请贫困'],prefix='是否在校申请贫困')
source_data['是否在校申请贫困'] = abs(source_data['是否在校申请贫困'] -1)%254
# print(source_data['是否在校申请贫困'])

# print(pd.unique(source_data['生活费']))
living_expenses_extremefilter = ['0','1000多','父母给','不知','500万','没算过','不定','不造','不固定','100','1','20000000','包括','不知道','100000','20000000000000000' '200']
living_expenses_correctablefilter = {'两千':2000,'2000➕':2000,'1500-':1500,'1200-1500':1350,'2k':2000,'1500元':1500}
source_data['生活费'] = source_data['生活费'].apply(lambda x : living_expenses_correctablefilter[x] if x in living_expenses_correctablefilter else x )
source_data['生活费'] = source_data['生活费'].apply(lambda x : np.nan if x in living_expenses_extremefilter else int(x) )
# print(pd.unique(source_data['生活费']))
source_data.dropna(axis=0,how='any',inplace=True)

# print(pd.unique(source_data['基础消费']))
basic_consumption_extremefilter = ['吃饭','不知道，看心情','999999999']
basic_consumption_correctablefilter = {'500-600':550,'两千':2000,'700-850':775,'1500➕':1500,'1.5k':1500,'850元' :850,'1200+':1200}
source_data['基础消费'] = source_data['基础消费'].apply(lambda x : basic_consumption_correctablefilter[x] if x in basic_consumption_correctablefilter else x )
source_data['基础消费'] = source_data['基础消费'].apply(lambda x : np.nan if x in basic_consumption_extremefilter else int(x) )
source_data.dropna(axis=0,how='any',inplace=True)
(row, col) = source_data.shape
print((row, col))


# source_data.to_excel(r'data\数据清洗.xlsx')
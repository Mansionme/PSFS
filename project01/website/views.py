from django.shortcuts import render,redirect
import numpy as np
import pandas as pd
import xlwt,xlrd
import joblib
import os
import tensorflow as tf
# load model
DLmodel = tf.keras.models.load_model(r'DLmodel')
LinearSVCmodel = joblib.load(filename="LinearSVCmodel.pkl")
RFmodel = joblib.load(filename="RFmodel.pkl")
XGBmodel = joblib.load(filename="XGBmodel.pkl")
def home(request):
    return render(request,'home.html')
def index(request):
    if request.method =='GET':
        return render(request,"index.html")
    if request.method =='POST': #获取数据
        nocomedu = request.POST.get("Noncomedu")
        disease = request.POST.getlist("disease")
        poverty_areas= request.POST.getlist("poverty areas")
        labor = request.POST.get("labor")
        Emergencies = request.POST.getlist("Emergencies")
        Family_debt = request.POST.get("Family debt")
        monthly_income = request.POST.get("monthly income")
        poor = ['1']
        expenses = request.POST.get("expenses")
        consumption = request.POST.get("consumption")
        shopping = request.POST.getlist("shopping")
        Food_and_clothing = request.POST.getlist("Food_and_clothing")
        entertainment = request.POST.getlist("entertainment")
        in_love = request.POST.getlist("in_love")
        cosmetic = request.POST.getlist("cosmetic")
        f_vocation = request.POST.getlist("f_vocation")
        m_vocation = request.POST.getlist("m_vocation")
        a = [disease,poverty_areas,Emergencies,poor,shopping,Food_and_clothing,entertainment,in_love,cosmetic,f_vocation]
        b = []
        for x in a:    #数据清洗
            b.append(int(x[0]))
        b.insert(0,nocomedu)
        b.insert(3,labor)
        b.insert(5,Family_debt)
        b.insert(6,monthly_income)
        b.insert(8,expenses)
        b.insert(9,consumption)
        wb = xlwt.Workbook()
        wt = wb.add_sheet("sheet1",cell_overwrite_ok=True)
        wt.write(0,1,'家庭受非义务教育人数')
        wt.write(0,2,'是否重大疾病')
        wt.write(0,3,'贫困地区')
        wt.write(0,4,'劳动力人口')
        wt.write(0,5,'突发事件')
        wt.write(0,6,'家庭负债')
        wt.write(0,7,'人均月收入')
        wt.write(0,8,'是否在校申请贫困')
        wt.write(0,9,'生活费')
        wt.write(0,10,'基础消费')
        wt.write(0,11,'网购')
        wt.write(0,12,'温饱问题')
        wt.write(0,13,'娱乐')
        wt.write(0,14,'谈恋爱')
        wt.write(0,15,'护肤或者化妆品')
        wt.write(0,16,'务农')
        wt.write(0,17,'失地农民')
        wt.write(0,18,'失业')
        wt.write(0,19,'个体工商户')
        wt.write(0,20,'城镇农民工')
        wt.write(0,21,'其它')
        for x in range(len(b)-1):
            wt.write(1,x+1,b[x])
        wt.write(1,0,0)
        alist = [16,17,18,19,20,21]
        print(f_vocation)
        print(m_vocation)
        for x in alist:
            if(int(f_vocation[0]) == x):
                wt.write(1,x,1)
            else:
                wt.write(1,x,0)
        for y in alist:
            if(int(m_vocation[0]) == y):
                wt.write(1,y,1)
        wb.save('data.xls')
        return redirect('/answer/')

def answer(request):
    data = pd.read_excel(r'data.xls')
    sample = data.loc[0]
    # x = sample.drop(['Unnamed: 0','是否在校申请贫困'],axis=1)
    x = sample.drop(['Unnamed: 0','是否在校申请贫困']).values
    x = np.array(x).reshape(-1, x.shape[0])
    # classify the sample(make a judgement)
    #-------未使用模型时做的假数据 测试用
    # y_pre = 0.6
    # return render(request,"answer.html",{'flag':1,'n_probability':y_pre,'p_probability':(1-y_pre)})
    lambda1 = lambda2 = lambda3 = lambda4 = 0.25
    y_pre =   (lambda1 * LinearSVCmodel.predict_proba(x)[0][0]+ lambda2 * RFmodel.predict_proba(x)[0][0]+ 
         lambda3 * XGBmodel.predict_proba(x)[0][0] + lambda4 * DLmodel.predict(x)[0][0])
    if (y_pre > .5):
        y_pre1 = "%.2f%%" % (y_pre * 100)
        y_pre2 = "%.2f%%" % ((1-y_pre) * 100)
        return render(request,'answer.html',{'flag':1,'n_probability':y_pre1,'p_probability':y_pre2})
    else:
        return render(request,'answer.html',{'flag':0,'n_probability':y_pre1,'p_probability':y_pre2})

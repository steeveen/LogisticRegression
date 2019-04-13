# -*- coding: utf-8 -*-
'''
        司马懿：“善败能忍，然厚积薄发”
                                    ——李叔说的
code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
          --┃      ☃      ┃--
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗II━II┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
 @Belong = 'LogisticRegression'  @MadeBy = 'PyCharm'
 @Author = 'steven'   @DateTime = '2019/4/10 15:55'
'''
import numpy as np
from LogisticRegression import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data=load_iris()
x=data['data']
y=(data['target']>0).astype(np.int8)
trainX,testX,trainY,testY=train_test_split(x,y,test_size=0.2,shuffle=True)
trainX=trainX.T
testX=testX.T
trainY=trainY.reshape(1,-1)
testY=testY.reshape(1,-1)
featureNum=4
lg=LogisticRegression(featureNum,{'lr':0.00001})
lg.fit(trainX,trainY,2000)
lg.eva(testX,testY)

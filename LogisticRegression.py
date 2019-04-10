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
 @Author = 'steven'   @DateTime = '2019/4/10 14:43'
'''
import numpy as np
from Activation import *
from LossFun import *
from Metrics import *

class LogisticRegression:
    def __init__(self, inputSize, config=None):
        '''
        :param inputSize:样本的特征维度
        '''
        self.inputSize = inputSize
        self.w = np.random.random((self.inputSize,1))#(m,1)
        self.b = np.random.random()
        self.config(config)

    def config(self, config):
        # TODO 实现对分类器的配置
        '''
        对分类器进行基本的配置
        :param config: 需要配置的属性（损失函数，优化算法选择等）
        :return:
        '''
        self.act = sigmoid
        self.lossFun=crossEntropy
        self.lr=0.001

    def fit(self, x, y,epoch=20):
        '''
        训练时调用
        :param x:一个批次的样本,二维矩阵，shape=(样本数量，特征维数)
        :param y:样本对应的类别，使用[0,1]形式,shape=(1,样本数量)
        :return:
        '''
        if np.ndim(x) != 2:
            raise Exception('输入数据必须为二维矩阵')
        if np.shape(x)[0] != self.inputSize:
            raise Exception('特征维度不匹配')

        for _ in range(epoch):
            # TODO 现在一个epoch就是i一个batch，即采用 梯度下降法
            self.trainOneBatch(x,y)
            print('%d/%d is over ,the loss is %f'%(_,epoch,np.sum(self.lossFun(y,self.predict(x)))))

    def trainOneBatch(self,x,y):
        #训练一个Batch
        self.z = np.dot(self.w.T, x) + self.b  # (1,n)
        self.a = sigmoid(self.z)  # (1,n)
        self.L = self.lossFun(y, self.a)  # (1,n)
        self.da = (1 - y) / (1 - self.a) - y / self.a  # (1,n)
        # self.dz=self.da*(self.a*(1-self.a))#(1,n)
        self.dz = self.a - y  # (1,n)
        self.dw = np.dot( x, self.dz.T) / self.inputSize
        self.db = np.sum(self.dz) / self.inputSize

        self.w += self.lr * self.dw
        self.b += self.lr * self.db
    def predict(self,x):
        return self.act(np.dot( self.w.T,x)+self.b)
    def eva(self,x,y):
        y_p=self.predict(x)
        loss=np.sum(self.lossFun(y,y_p))
        print('loss is '+str(loss))
        print('acc is '+str(acc(y,(y_p>0.5).astype(np.int8))))
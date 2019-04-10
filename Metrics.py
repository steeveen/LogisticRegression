# -*- coding: utf-8 -*-
'''评价标准(基于numpy)
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
 @Author = 'steven'   @DateTime = '2019/4/10 15:47'
'''
import numpy as np
def acc(y_t,y_p):
    return np.sum(np.logical_not(np.logical_xor(y_t,y_p)))/np.size(y_t)
def recall(y_t,y_p):
    return np.sum(np.logical_and(y_t,y_p))/np.sum(y_t)
def precision(y_t,y_p):
    return np.sum(np.logical_and(y_t,y_p)).np.sum(y_p)
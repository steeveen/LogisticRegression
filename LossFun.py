# -*- coding: utf-8 -*-
'''损失函数（基于numpy）
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
 @Author = 'steven'   @DateTime = '2019/4/10 15:11'
'''
import numpy as np
def crossEntropy(y_t,y_p):
    '''
    二类交叉熵
    :param y_t:分类器输出值
    :param y_p: 标签值
    :return:
    '''
    return -y_t*np.log(y_p)-(1-y_t)*np.log(1-y_p)

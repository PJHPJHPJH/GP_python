import numpy as np
import random
import pandas as pd
from numpy import *
from random import normalvariate  # 正态分布
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def loadData(base_dir):
    '''
    load ori_data
    Returns: csv_data
    '''

    movie_columns = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(base_dir + "movies.dat", sep='::', header=None, names=movie_columns, engine='python')

    rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(base_dir + "ratings.dat", sep='::', header=None, names=rating_columns, engine='python')

    user_columns = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_csv(base_dir + "users.dat", sep='::', header=None, names=user_columns, engine='python')

    # pd.merge(A,B)
    # 没有指定连接列.merge()默认采取两个DataFrame中都有的列做为连接键
    # 连接方式默认为保留两个DataFrame中都有数据的行
    data = pd.merge(ratings, movies)
    data = pd.merge(data, users)
    return data




# 处理数据
def preprocessData(data, ratio):

    #获得标签，将评分低于3分的样本标记为负样本，高于三分的标记为正样本
    label = data['rating'].map(lambda x: 1 if x > 3 else -1)

    features = ['genres', 'gender', 'age']
    data = data[features]
    #onehot
    #pd.get_dummies()将文本标签转换成onehot编码
    data = pd.get_dummies(data)

    #用于划分train data 和test data
    num = int(data.shape[0] * ratio)
    train = data[:num]
    train_label = label[:num]

    test = data[num:]
    test_label = label[num:]

    #返回划分好的数据集
    return train, train_label, test, test_label




def sigmoid(inx):
    '''
        激活函数
    '''
    return 1.0 / (1 + np.exp(-inx))


# 训练FM模型
def FM(dataMatrix, classLabels, k, iter, alpha):
    '''
    :param dataMatrix:  特征矩阵
    :param classLabels: 标签矩阵
    :param k:           v的维数
    :param iter:        迭代次数
    :param alpha:       学习率
    :return:            常数项w_0, 一阶特征系数w, 二阶交叉特征系数v
    '''
    # dataMatrix用的是matrix, classLabels是列表
    m, n = shape(dataMatrix)  # 矩阵的行列数，即样本数m和特征数n

    # 初始化参数
    w = zeros((n, 1))  # 一阶特征的系数
    w_0 = 0  # 常数项
    # 即生成辅助向量(n*k)，用来训练二阶交叉特征的系数
    v = normalvariate(0, 0.2) * ones((n, k))


    for it in range(iter):
        for x in range(m):  # 随机优化，每次只使用一个样本
            # 二阶项的计算
            inter_1 = dataMatrix[x] * v  # 每个样本(1*n)x(n*k),得到k维向量（FM化简公式大括号内的第一项）
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # 二阶交叉项计算，得到k维向量（FM化简公式大括号内的第二项）
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.  # 二阶交叉项计算完成（FM化简公式的大括号外累加）


            #为什么P是二维
            p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出，即FM的全部项之和
            tmp = 1 - sigmoid(classLabels[x] * p[0, 0])  # tmp迭代公式的中间变量，便于计算
            w_0 = w_0 + alpha * tmp * classLabels[x]

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] + alpha * tmp * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j] + alpha * tmp * classLabels[x] * (
                                dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

        # 计算损失函数的值
        if it % 10 == 0:
            loss = getLoss(getPrediction(mat(dataMatrix), w_0, w, v), classLabels)
            print("第{}次迭代后的损失为{}".format(it, loss))

    return w_0, w, v


# 损失函数
def getLoss(predict, classLabels):
    m = len(predict)
    loss = 0.0
    for i in range(m):
        loss -= log(sigmoid(predict[i] * classLabels[i]))
    return loss


# 预测
def getPrediction(dataMatrix, w_0, w, v):
    m = np.shape(dataMatrix)[0]
    result = []
    for x in range(m):
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  # multiply对应元素相乘
        # 完成交叉项
        interaction = np.sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出
        pre = sigmoid(p[0, 0])
        result.append(pre)
    return result


# 评估预测的准确性
def getAccuracy(predict, classLabels):
    m = len(predict)
    allItem = 0
    error = 0
    for i in range(m):  # 计算每一个样本的误差
        allItem += 1
        if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
            error += 1
        elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
            error += 1
        else:
            continue

    return float(error) / allItem


if __name__ == '__main__':
    data_file = "ml-1m/"
    #从外部加载数据
    Data = loadData(data_file)
    #拆分数据集
    dataTrain, labelTrain, dataTest, labelTest = preprocessData(Data, 0.8)
    #记录开始时间
    date_startTrain = datetime.now()

    print("开始训练")

    #获得训练完成的最优参数
    # w_0, w, v = FM(mat(dataTrain), labelTrain, 4, 100, 0.001)
    w_0, w, v = FM(mat(dataTrain), labelTrain, 2, 5, 0.001)
    print("w_0:", w_0)
    print("w:", w)
    print("v:", v)

    #train data 验证
    predict_train_result = getPrediction(mat(dataTrain), w_0, w, v)  # 得到训练的准确性
    #训练误差
    print("训练准确性为：%f" % (1 - getAccuracy(predict_train_result, labelTrain)))
    #训练结束时间
    date_endTrain = datetime.now()
    print("训练用时为：%s" % (date_endTrain - date_startTrain))

    print("开始测试")
    #测试误差
    predict_test_result = getPrediction(mat(dataTest), w_0, w, v)  # 得到训练的准确性
    print("测试准确性为：%f" % (1 - getAccuracy(predict_test_result, labelTest)))








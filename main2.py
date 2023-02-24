# 载入第三方库
from flask import Flask, make_response
from flask import jsonify
import numpy
import pandas as pd
# import tensorflow as tf
# 解决版本兼容性问题
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import os
import random
from collections import defaultdict
from datetime import datetime





app = Flask(__name__)




# 载入数据

def load_data(data_path):
    '''
        data_path 数据存放的路径
    '''

    data = pd.read_csv(data_path)
    print(data.shape)

    # 改变索引顺序
    data = data.loc[:, ['visitorid', 'itemid', 'event', 'timestamp']]
    data = numpy.array(data)

    # 创建一个set的改进版本，在不存在key时返回默认值
    user_ratings = defaultdict(set)

    # 记录用户和商品的数量，用于确定训练矩阵的大小
    max_u_id = -1
    max_i_id = -1

    # 读取数据
    for m in range(data.shape[0]):
        # 数据格式：user_id item_id rating timestamp
        # 训练时只用到user和item
        #         u, i, _, _ = line.split("\t")
        u = numpy.array(data[m][0])

        u = int(u)
        i = numpy.array(data[m][1])
        i = int(i)
        # 构建user-item的交互表
        # 若user点击过item则加入user_ratings中
        user_ratings[u].add(i)
        max_u_id = max(u, max_u_id)
        max_i_id = max(i, max_i_id)
    print("max_u_id:", max_u_id)
    print("max_i_id:", max_i_id)
    return max_u_id, max_i_id, user_ratings


# 数据预处理
# 产生测试集
def generate_test(user_ratings):
    '''
        user_ratings load_data()的返回值
    '''
    user_test = dict()
    # 取出相应的键值对
    for u, i_list in user_ratings.items():
        # 每个user从自己的交互集中取出一个item，构成验证样本
        user_test[u] = random.sample(user_ratings[u], 1)[0]
    return user_test


# 数据预处理
# 产生训练集
def generate_train_batch(user_ratings, user_ratings_test,
                         item_count, batch_size=512):
    '''
        user_ratings load_data()的返回值
        user_ratings_test generate_test的返回值
        item_count item总数量
        batch_size 每次迭代的样本数量
        i 表示对于user的正反馈
        j 表示对于user的负反馈
        要求i>uj,即对于用户u来说，i排名在j前面
        return 测试集((u,i,j)的集合，ndarray形式)
    '''
    t = []
    for b in range(batch_size):
        # 随机取出一个user
        u = random.sample(user_ratings.keys(), 1)[0]
        # 从user对应的集合中取出一个item
        i = random.sample(user_ratings[u], 1)[0]

        # 如果取出的item在测试集中，则另取一个
        while i == user_ratings_test[u]:
            i = random.sample(user_ratings[u], 1)[0]

        # 随机取一个样本，该样本不在user对应的集合中
        j = random.randint(1, item_count)
        while j in user_ratings[u]:
            j = random.randint(1, item_count)

        # 产生满足训练条件的(u,i,j)三元组
        t.append([u, i, j])
    return numpy.asarray(t)

#产生测试集
def generate_test_batch(user_ratings, user_ratings_test, item_count):
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        # 将所有不在user_ratings[u]的item全部作为测试集
        for j in range(1, item_count+1):
            if not (j in user_ratings[u]):
                t.append([u, i, j])
        yield numpy.asarray(t)


# 定义数据流图
def bpr_mf(user_count, item_count, hidden_dim):
    '''
        m:user_count
        n:item_count
        X = W · H , W:mxk, H:kxn
        hidden_dim BPR算法的隐藏维度k
    '''

    # tf.placeholder(dtype, shape=None, name=None)
    # 定义tensorflow能处理的标准形式,相当于占位符，
    # 在session.run()中使用feed_dict中传入真实的值
    # dtype 数据类型
    # shape 数据格式，默认为[None]，表示一维向量
    # name 名称
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])

    # 使用设备
    with tf.device("/gpu:0"):
        # tf.get_variable(name, shape, initializer)
        # 产生新的tensorflow变量

        # user_emb_w 等同于W
        # item_emb_w 等同于H
        user_emb_w = tf.get_variable("user_emb_w", [user_count + 1, hidden_dim],
                                     initializer=tf.random_normal_initializer(0, 0.1))
        item_emb_w = tf.get_variable("item_emb_w", [item_count + 1, hidden_dim],
                                     initializer=tf.random_normal_initializer(0, 0.1))

        # 数据映射
        u_emb = tf.nn.embedding_lookup(user_emb_w, u)
        i_emb = tf.nn.embedding_lookup(item_emb_w, i)
        j_emb = tf.nn.embedding_lookup(item_emb_w, j)

    # MF predict: u_i > u_j
    # tf.reduce_sum(tensor, axis, keep_dims)作用是按一定方式计算张量中元素之和
    # keep_dims =True 维持张tensor的维度
    x = tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keep_dims=True)

    # AUC for one user:
    # reasonable iff all (u,i,j) pairs are from the same user
    #
    # average AUC = mean( auc for each user in test set)
    mf_auc = tf.reduce_mean(tf.to_float(x > 0))

    l2_norm = tf.add_n([
        tf.reduce_sum(tf.multiply(u_emb, u_emb)),
        tf.reduce_sum(tf.multiply(i_emb, i_emb)),
        tf.reduce_sum(tf.multiply(j_emb, j_emb))
    ])

    # 正则化参数
    regulation_rate = 0.0001

    # 损失函数
    bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))

    # 随机梯度下降更新参数
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(bprloss)
    return u, i, j, mf_auc, bprloss, train_op




@app.route("/")
def home():
    data_path = "./event_100000.csv"
    user_count, item_count, user_ratings = load_data(data_path)
    user_ratings_test = generate_test(user_ratings)


    time_startTrain = 0
    time_endTrain = 0


    # 开始训练
    with tf.Graph().as_default(), tf.Session() as session:
        # 获得所有参数和函数的定义
        u, i, j, mf_auc, bprloss, train_op = bpr_mf(user_count,
                                                    item_count, 2)

        # 对变量进行初始化，变量运行前必须做初始化操作
        session.run(tf.initialize_all_variables())

        # 记录开始训练时间
        time_startTrain = datetime.now()

        # 迭代次数
        for epoch in range(0, 1):
            _batch_bprloss = 0
            for k in range(1, 200):  # uniform samples from training set
                # 获得训练集
                uij = generate_train_batch(user_ratings, user_ratings_test, item_count)

                _bprloss, _train_op = session.run([bprloss, train_op],
                                                  feed_dict={u: uij[:, 0], i: uij[:, 1], j: uij[:, 2]})

                _batch_bprloss += _bprloss

            print("epoch: ", epoch)
            print("bpr_loss: ", _batch_bprloss / k)
            print("_train_op")

            user_count = 0
            _auc_sum = 0.0

            # each batch will return only one user's auc
            for t_uij in generate_test_batch(user_ratings,
                                             user_ratings_test, item_count):
                _auc, _test_bprloss = session.run([mf_auc, bprloss],
                                                  feed_dict={u: t_uij[:, 0], i: t_uij[:, 1], j: t_uij[:, 2]}
                                                  )
                user_count += 1
                _auc_sum += _auc
            print("test_loss: ", _test_bprloss, "test_auc: ", _auc_sum / user_count)
            print("")
        # 训练完成的时间
        time_endTrain = datetime.now()
        print("训练用时为：%s" % (time_endTrain - time_startTrain))

        # 获得所有参数
        variable_names = [v.name for v in tf.trainable_variables()]

        # 输出的W,H矩阵分别在values[0]和values[1]中
        values = session.run(variable_names)
        for k, v in zip(variable_names, values):
            print("Variable: ", k)
            print("Shape: ", v.shape)
            print(v)

    session1 = tf.Session()

    u1_dim = tf.expand_dims(values[0][0], 0)

    u1_all = tf.matmul(u1_dim, values[1], transpose_b=True)
    result_1 = session1.run(u1_all)
    print(result_1)



    result = {}

    print("以下是给用户0的推荐：")
    print("商品ID\t综合得分")
    p = numpy.squeeze(result_1)
    p[numpy.argsort(p)[:-5]] = 0
    for index in range(len(p)):
        if p[index] != 0:
            print(index, p[index])
            result[str(index)] = str(p[index])
    result[str("time:")] = str((time_endTrain - time_startTrain))
    print(result)

    return jsonify(result)



# 通过后置钩子函数，解决跨域问题
@app.after_request
def func_res(resp):
    res = make_response(resp)
    res.headers['Access-Control-Allow-Origin'] = '*'
    res.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    res.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return res


if __name__ == "__main__":
    app.run(debug=True)

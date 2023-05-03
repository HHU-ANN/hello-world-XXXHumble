# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    beta = np.linalg.inv(X.T.dot(X) + 1e-12 * np.identity(X.shape[1])).dot(X.T).dot(y)
    return beta
    
def lasso(data):
    n_features = X.shape[1]
    beta = np.zeros(n_features)
    for i in range(max_iter):
        # 计算梯度
        gradient = X.T.dot(X.dot(beta) - y) + 1e-12 * np.sign(beta)
        # 更新权重
        beta -= lr * gradient
    return beta

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

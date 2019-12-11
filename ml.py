import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.optimize import minimize

# 线性回归
# data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
# # data.plot.scatter('population', 'profit', label='population vs profit')
# # plt.show()
#
# data.insert(0, 'ones', 1)
# X = data.iloc[:, 0:-1].values
# y = data.iloc[:, -1].values
# y = y.reshape(97, 1)
#
# def costFunction(x, y, theta):
#     inner = np.power(x @ theta - y, 2)
#     return np.sum(inner)/(2 * len(x))
#
# def gradientDescent(x, y, theta, alpha, iters):
#     costs = []
#     for i in range(iters):
#         theta = theta - alpha * (x.T @ (x @ theta - y))/len(x)
#         cost = costFunction(x, y, theta)
#         costs.append(cost)
#     return costs, theta
#
# theta = np.zeros((2, 1))
# alpha = 0.02
# iters = 2000
#
#
# costs, theta = gradientDescent(X, y, theta, alpha, iters)
#
# # fig, ax = plt.subplots()
# # ax.plot(np.arange(iters), costs, 'r')
# # ax.set(xlabel='iters', ylabel='cost', title='iters vs cost')
# # plt.show()
#
# fig, ax = plt.subplots()
# ax.scatter(X[:, -1], y, label='training data')
#
# x = np.linspace(X.min(), X.max(), 100)
# y_ = theta[0, 0] + theta[1, 0]*x
# ax.plot(x, y_, 'r', label='predict')
# ax.set(xlabel='population', ylabel='profit')
# ax.legend()
# plt.show()

# data = pd.read_csv('ex1data2.txt', names=['size', 'bedrooms', 'price'])
#
# # 特征归一化
# def normallize_feature(data):
#     return (data - data.mean())/data.std()
#
# data = normallize_feature(data)
#
# # data.plot.scatter('size', 'price', label='size')
# # plt.show()
#
# data.insert(0, 'ones', 1)
# X = data.iloc[:, 0:-1].values
# y = data.iloc[:, -1].values
# y = y.reshape(47, 1)
#
# def costFunction(X, y, theta):
#     inner = np.power(X @ theta - y, 2)
#     return np.sum(inner)/(2*len(X))
#
#
# def gradientDescent(X, y, theta, alpha, iters):
#     costs = []
#     for i in range(iters):
#         theta = theta - alpha * X.T @ (X @ theta - y) / len(X)
#         cost = costFunction(X, y, theta)
#         costs.append(cost)
#     return theta, costs
#
#
# theta = np.zeros((3, 1))
# iters = 2000
# alpha = 0.03
#
#
# theta, costs = gradientDescent(X, y, theta, alpha, iters)
#
# # fig, ax = plt.subplots()
# # for item in alpha:
# #     _, costs = gradientDescent(X, y, theta, item, iters)
# #     ax.plot(np.arange(iters), costs, label=item)
# # ax.set(xlabel='iters', ylabel='costs', title='costs vs iters')
# # ax.legend()
# # plt.show()
#
#
# # 正规方程
# def normalEquation(X, y):
#     return np.linalg.inv(X.T@X)@X.T@y
#
# theta = normalEquation(X, y)

# 逻辑回归

# 逻辑回归
# data = pd.read_csv('ex2data1.txt', names=['Exam 1', 'Exam 2', 'Accepted'])

# fig, ax = plt.subplots()
# ax.scatter(data[data['Accepted'] == 1]['Exam 1'], data[data['Accepted'] == 1]['Exam 2'], c='b', label='y=1', marker='o')
# ax.scatter(data[data['Accepted'] == 0]['Exam 1'], data[data['Accepted'] == 0]['Exam 2'], c='r', label='y=0', marker='x')
# ax.legend()
# ax.set(xlabel='exam1', ylabel='exam2')
# plt.show()

# data.insert(0, 'ones', 1)
# X = data.iloc[:, 0:-1].values
# y = data.iloc[:, -1].values
# y = y.reshape(100, 1)
#
# def sigmoid(z):
#     return 1/(1 + np.exp(-z))
#
# def costFunction(X, y, theta):
#     S = sigmoid(X@theta)
#
#     first = y*np.log(S)
#     second = (1 - y)*np.log(1 - S)
#
#     return -np.sum(first + second)/len(X)
#
#
# def gradientDescent(X, y, theta, alpha, iters):
#     costs = []
#     for i in range(iters):
#         S = sigmoid(X @ theta)
#         theta = theta - (alpha/len(X))*X.T@(S - y)
#         cost = costFunction(X, y, theta)
#         costs.append(cost)
#     return theta, costs
#
# theta = np.zeros((3, 1))
# alpha = 0.004
# iters = 200000
#
# theta_final, costs = gradientDescent(X, y, theta, alpha, iters)
#
# fig, ax = plt.subplots()
# ax.plot(np.arange(iters), costs)
# ax.set(xlabel='iters', ylabel='costs', title='costs vs iters')
# plt.show()


# def predict(X, theta):
#     prob = sigmoid(X@theta)
#     return [1 if x >= 0.5 else 0 for x in prob]
#
# y_ = np.array(predict(X, theta_final))
# y_pre = y_.reshape(len(y_), 1)
#
# acc = np.mean(y_pre == y)
# print(acc)
#
# coef1 = -theta_final[0, 0]/theta_final[2, 0]
# coef2 = -theta_final[1, 0]/theta_final[2, 0]
#
# x = np.linspace(20, 100, 200)
# f = coef1 + coef2*x
#
# fig, ax = plt.subplots()
# ax.scatter(data[data['Accepted'] == 1]['Exam 1'], data[data['Accepted'] == 1]['Exam 2'], c='b', label='y=1', marker='o')
# ax.scatter(data[data['Accepted'] == 0]['Exam 1'], data[data['Accepted'] == 0]['Exam 2'], c='r', label='y=0', marker='x')
# ax.legend()
# ax.plot(x, f, 'g')
# ax.set(xlabel='exam1', ylabel='exam2')
# plt.show()

# data = pd.read_csv('ex2data2.txt', names=['Test1', 'Test2', 'Accepted'])

# fig, ax = plt.subplots()
# ax.scatter(data[data['Accepted'] == 1]['Test1'], data[data['Accepted'] == 1]['Test2'], c='b', marker='o', label='y=1')
# ax.scatter(data[data['Accepted'] == 0]['Test1'], data[data['Accepted'] == 0]['Test2'], c='r', marker='x', label='y=0')
# ax.legend()
# ax.set(xlabel='Test1', ylabel='Test2')
# plt.show()


# 特征映射
# def feature_mapping(x1, x2, power):
#     data = {}
#
#     for i in np.arange(power+1):
#         for j in np.arange(i+1):
#             data['F{}{}'.format(i-j, j)] = np.power(x1, i-j)*np.power(x2, j)
#     return pd.DataFrame(data)
#
#
# x1 = data['Test1']
# x2 = data['Test2']
#
# data2 = feature_mapping(x1, x2, 6)
#
# # 数据处理
# X = data2.values
# y = data.iloc[:, -1].values
# y = y.reshape(len(y), 1)
#
# # 正则化
# def sigmoid(z):
#     return 1/(1 + np.exp(-z))
#
#
# def costFunction(X, y, theta, lamda):
#     sig = sigmoid(X@theta)
#
#     first = y*np.log(sig)
#     second = (1-y)*np.log(1-sig)
#     reg = lamda/(2*len(X))*np.sum(np.power(theta[1:], 2))
#
#     return -np.sum(first + second)/len(X) + reg
#
#
# def gradientDescent(X, y, theta, alpha, iters, lamda):
#     costs = []
#
#     for i in range(iters):
#         reg = theta[1:]*(lamda/len(X))
#         reg = np.insert(reg, 0, values=0, axis=0)
#
#         theta = theta - (X.T@(sigmoid(X@theta)-y))*alpha/len(X) - reg
#         cost = costFunction(X, y, theta, lamda)
#         costs.append(cost)
#
#     return theta, costs
#
#
# alpha = 0.001
# iters = 200000
# lamda = 0.001
# theta = np.zeros((28, 1))
#
# theta_final, costs = gradientDescent(X, y, theta, alpha, iters, lamda)
#
#
# def predict(X, theta):
#     prob = sigmoid(X@theta)
#     return [1 if x >= 0.5 else 0 for x in prob]
#
# y_ = np.array(predict(X, theta_final))
# y_pre = y_.reshape(len(y_), 1)
#
# acc = np.mean(y_pre == y)
# print(acc)

# 逻辑回归多分类问题
# data = sio.loadmat('ex3data1.mat')
# X = data['X']
# y = data['y']

# image = X[1451, :]
# image = image.reshape(20, 20)
# fig, ax = plt.subplots(figsize=(1, 1))
# ax.imshow(image)
# plt.xticks([])
# plt.yticks([])
# plt.show()

# def sigmoid(z):
#     return 1/(1 + np.exp(-z))
#
#
# def costFunction(theta, X, y, lamda):
#     S = sigmoid(X@theta)
#
#     first = y*np.log(S)
#     second = (1-y)*np.log(1-S)
#
#     reg = theta[1:] @ theta[1:] * (lamda / (2*len(X)))
#
#     return -np.sum(first + second) / len(X) + reg
#
#
# def Gradient_reg(theta, X, y, lamda):
#     reg = theta[1:]*(lamda/len(X))
#     reg = np.insert(reg, 0, values=0, axis=0)
#
#     first = (X.T@(sigmoid(X@theta) - y))/ len(X)
#
#     return first + reg
#
#
# X = np.insert(X, 0, values=1, axis=1)
# y = y.flatten()
#
# def one_vs_all(X,y,lamda,K):
#
#     n = X.shape[1]
#
#     theta_all = np.zeros((K, n))
#
#     for i in range(1, K+1):
#         theta_i = np.zeros(n,)
#
#         res = minimize(fun=costFunction,
#                        x0=theta_i,
#                        args=(X, y == 1, lamda),
#                        method='TNC',
#                        jac=Gradient_reg)
#         theta_all[i-1, :] = res.x
#
#     return theta_all


# # BP-神经网络
# import numpy as np
# import scipy.io as sio
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize
#
#
#
# data = sio.loadmat('ex4data1.mat')
# raw_X = data['X']
# raw_y = data['y']
#
# X = np.insert(raw_X, 0, values=1, axis=1)
#
# # 对y进行one_hot编码
# def one_hot_encode(raw_y):
#     result = []
#
#     for i in raw_y:#1-10
#         y_temp = np.zeros(10)
#         y_temp[i-1] = 1
#
#         result.append(y_temp)
#
#     return np.array(result)
#
# y = one_hot_encode(raw_y)
#
#
# theta = sio.loadmat('ex4weights.mat')
# theta1, theta2 = theta['Theta1'], theta['Theta2']
#
# # 序列化权重参数
# def serialize(a, b):
#     return np.append(a.flatten(), b.flatten())
#
#
# theta_serialize = serialize(theta1, theta2)
#
#
# # 解序列化权重参数
# def deserialize(theta_serialize):
#     theta1 = theta_serialize[:25*401].reshape(25, 401)
#     theta2 = theta_serialize[25*401:].reshape(10, 26)
#     return theta1, theta2
#
# theta1, theta2 = deserialize(theta_serialize)
#
#
#
# # 前向传播
# def sigmoid(Z):
#     return 1/(1 + np.exp(-Z))
#
#
# def feed_forward(theta_serialize, X):
#     theta1, theta2 = deserialize(theta_serialize)
#     a1 = X
#     z2 = a1@theta1.T
#     a2 = sigmoid(z2)
#     a2 = np.insert(a2, 0, values=1, axis=1)
#     z3 = a2@theta2.T
#     h = sigmoid(z3)
#     return a1, z2, a2, z3, h
#
# # 不带正则化的损失函数
# def cost(theta_serialize, X, y):
#     a1, z2, a2, z3, h = feed_forward(theta_serialize, X)
#     J = -np.sum(y*np.log(h)+(1-y)*np.log(1-h)) / len(X)
#     return J
#
#
# def reg_cost(theta_serialize, X, y, lamda):
#     sum1 = np.sum(np.power(theta1[:, 1:], 2))
#     sum2 = np.sum(np.power(theta2[:, 1:], 2))
#     reg = (sum1 + sum2)*lamda/(2*len(X))
#     return reg + cost(theta_serialize, X, y)
#
# lamda = 1


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.io import loadmat
# from scipy.optimize import minimize
#
#
# data = loadmat('ex5data1.mat')
# # print(data.keys())
#
# # 训练集
# X_train, y_train = data['X'], data['y']
# # print(X_train.shape, y_train.shape)
#
# # 验证集
# X_val, y_val = data['Xval'], data['yval']
# # print(X_val.shape, y_val.shape)
#
# # 测试集
# X_test, y_test = data['Xtest'], data['ytest']
# # print(X_test.shape, y_test.shape)
#
#
# X_train = np.insert(X_train, 0, 1, axis=1)
# X_val = np.insert(X_val, 0, 1, axis=1)
# X_test = np.insert(X_test, 0, 1, axis=1)
#
# def plot_data():
#     fig, ax = plt.subplots()
#     ax.scatter(X_train[:, -1], y_train)
#     ax.set(xlabel='change in water level(x)',
#            ylabel='water flowing out og the dam(y)')
#     plt.show()
#
# # plot_data()
#
# # 损失函数
# def reg_costFunction(theta, X, y, lamda):
#     cost = np.sum(np.power((X@theta-y.flatten()), 2))
#     reg = theta[1:]@theta[1:]*lamda
#
#     return (cost+reg)/(2*len(X))
#
# theta = np.ones(X_train.shape[1])
# lamda = 1
# # print(reg_costFunction(theta, X_train, y_train, lamda))
#
#
# # 梯度
# def reg_gradient(theta, X, y, lamda):
#     grad = (X@theta-y.flatten())@X
#     reg = lamda*theta
#     reg[0] = 0
#
#     return (grad + reg)/len(X)
#
#
# def train_model(X, y, lamda):
#     theta = np.ones(X.shape[1])
#
#     res = minimize(fun=reg_costFunction,
#                    x0=theta,
#                    args=(X, y, lamda),
#                    method='TNC',
#                    jac=reg_gradient)
#     return res.x
#
#
# theta_final = train_model(X_train, y_train, lamda=0)
#
# fig, ax = plt.subplots()
# ax.scatter(X_train[:, -1], y_train)
# ax.set(xlabel='change in water level(x)',
#        ylabel='water flowing out og the dam(y)')
# ax.plot(X_train[:, 1], X_train@theta_final, c='r')
# plt.show()




data = pd.read_csv('ex2data1.txt', names=['Exam1', 'Exam2', 'Accepted'])
# print(data.head())
# print(data.info())
# print(data.describe())
# fig, ax = plt.subplots()
# ax.scatter(data[data['Accepted'] == 1]['Exam1'],
#            data[data['Accepted'] == 1]['Exam2'],
#            c='b',
#            marker='x',
#            label='y=1')
# ax.scatter(data[data['Accepted'] == 0]['Exam1'],
#            data[data['Accepted'] == 0]['Exam2'],
#            c='r',
#            marker='o',
#            label='y=0')
# ax.set(xlabel='Exam1', ylabel='Exam2')
# ax.legend()
# plt.show()

X = data.iloc[:, :-1].values
X = np.insert(X, 0, 1, axis=1)
y = data.iloc[:, -1].values
y = y.reshape((100, 1))

def costFunction(X, y, theta):
    S = 1/(1+np.exp(-X@theta))
    left = y*np.log(S)
    right = (1-y)*np.log(1-S)
    return -np.sum(left+right)/len(X)

theta_init = np.zeros((3,1))

# cost_init = costFunction(X, y, theta_init)
# print(cost_init)


def gradientDescent(X, y, theta, alpha, iters):
    costs = []

    for i in range(iters):
        S = 1 / (1 + np.exp(-X @ theta))
        theta = theta-(alpha/len(X))*X.T@(S-y)
        cost = costFunction(X, y, theta)
        costs.append(cost)

    return theta, costs


alpha = 0.003
iters = 200000


theta_final, costs = gradientDescent(X, y, theta_init, alpha, iters)


print(theta_final)

# 一定要随便写点什么东西
# 我就是要看看git diff的作用
# 就是为了学习git
# ------------------------
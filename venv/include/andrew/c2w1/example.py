import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils   #第一部分，初始化
import reg_utils    #第二部分，正则化
import gc_utils     #第三部分，梯度校验
from reg import *


# 初始化
# plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'
#
# train_X, train_Y, test_X, test_Y = init_utils.load_dataset(is_plot=True)
#
# parameters = model(train_X, train_Y, initialization = "he",is_polt=True)
# print("训练集:")
# predictions_train = init_utils.predict(train_X, train_Y, parameters)
# print("测试集:")
# init_utils.predictions_test = init_utils.predict(test_X, test_Y, parameters)
# plt.title("Model with He initialization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 1.5])
# axes.set_ylim([-1.5, 1.5])
# init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

#正则化

train_X, train_Y, test_X, test_Y = reg_utils.load_2D_dataset(is_plot=True)
# parameters = model(train_X, train_Y,is_plot=True)
# print("训练集:")
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print("测试集:")
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
# plt.title("Model without regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

# L2
# parameters = model(train_X, train_Y, lambd=0.7,is_plot=True)
# print("使用正则化，训练集:")
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print("使用正则化，测试集:")
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
# plt.title("Model with L2-regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)

#dropout
parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3,is_plot=True)

print("使用随机删除节点，训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("使用随机删除节点，测试集:")
reg_utils.predictions_test = reg_utils.predict(test_X, test_Y, parameters)
plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)


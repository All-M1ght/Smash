import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import operator
from functools import reduce
# import sklearn.datasets
# import sklearn.linear_model
from lr_utils import *
from planar_utils import *

index = 10
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
plt.imshow(test_set_x_orig[index])
plt.show()
# print("train_set_y=" + str(train_set_y_orig[0][index]))

#X_flatten = X.reshape(X.shape [0]，-1).T ＃X.T是X的转置
#将训练集的维度降低并转置。
print(train_set_x_orig.shape[0])
train_set_x_flatten  = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
#将测试集的维度降低并转置。
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

#归一
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

parameters = nn_model(train_set_x, train_set_y_orig, n_h = 3, num_iterations=10000, print_cost=True)

Y_prediction_test = predict(parameters, test_set_x)
Y_prediction_train = predict(parameters, train_set_x)
# 打印训练后的准确性
print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - train_set_y_orig)) * 100), "%")
print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - test_set_y_orig)) * 100), "%")

mine = test_set_x_orig[index].reshape(1, -1).T
print(mine.shape)
Y_prediction = predict(parameters, mine)
print(Y_prediction.shape)
print(Y_prediction[0][0])
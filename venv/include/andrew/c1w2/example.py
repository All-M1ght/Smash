import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import *

index = 4
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

#这里加载的是真实的数据，请参见上面的代码部分。
d = model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
mine = test_set_x_orig[index].reshape(1, -1).T
print(mine.shape)
Y_prediction = predict(d["w"], d["b"], mine)
print(Y_prediction.shape)
print(Y_prediction[0][0])


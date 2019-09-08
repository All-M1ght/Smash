import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.framework import ops

import cnn_utils
from tensor import *
from scipy import ndimage
import scipy


# def predict(index):
#     x = X_train[index]
#     plt.imshow(x)
#     plt.show()
#     x = x.reshape(1, 64, 64, 3)
#     with tf.Session() as sess:
#         saver = tf.train.import_meta_graph("model.meta")  # 加载图
#         graph = tf.get_default_graph()
#         X = graph.get_tensor_by_name("X:0")  # 获取保存模型中的输入变量的tensor
#         Z3 = graph.get_tensor_by_name("fully_connected/BiasAdd:0")  # 获取softmax层的输入值
#         saver.restore(sess, tf.train.latest_checkpoint("./"))  # 加载模型
#         res = tf.nn.softmax(Z3)  # 执行softmax激活函数
#         f = sess.run(res, feed_dict={X: x})  # 进行预测
#         print("the value of prediction is ", np.argmax(f))
#

# def predict(index):
#     image = X_train_orig[index]
#     print("y = " + str(np.squeeze(Y_train_orig[:, index])))
#     plt.imshow(image)
#     plt.show()
#     x = np.expand_dims(np.array(X_train[index], dtype="float32"), 0)
#     y = np.expand_dims(Y_train[index], 0)
#     saver = tf.train.import_meta_graph('model.meta')
#     with tf.Session() as sess:
#         saver.restore(sess, tf.train.latest_checkpoint(''))
#         Probs = tf.nn.softmax(sess.run(Y, feed_dict={X: x, Y: y})).eval()
#     result = np.argmax(Probs)
#
#     return result


def predict(index):
    # x = X_test[index]
    x = index
    plt.imshow(x)
    plt.show()
    x = x.reshape(1, 64, 64, 3)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("cat/model.meta")  # 加载图
        saver.restore(sess, tf.train.latest_checkpoint("./cat"))  # 加载模型
        graph = tf.get_default_graph()
        # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        # for tensor_name in tensor_name_list:
        #     print(tensor_name, '\n')
        X = graph.get_tensor_by_name('Placeholder:0')  # 获取保存模型中的输入变量的tensor
        Z3 = graph.get_tensor_by_name("fully_connected/BiasAdd:0")  # 获取softmax层的输入值
        # res = tf.nn.softmax(Z3)  # 执行softmax激活函数
        res = tf.nn.sigmoid(Z3)  # 执行softmax激活函数
        # print(tf.Variable(res))
        f = sess.run(res, feed_dict={X: x})  # 进行预测
        print("the value of prediction is ", np.argmax(f))


np.random.seed(1)

X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = cnn_utils.load_cat_dataset()

X_train = X_train_orig/255.
X_test = X_test_orig/255.
# Y_train = Y_train_orig.T
# Y_test = Y_test_orig.T

# Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
# Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T
Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 2).T
Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 2).T
#
# _, _, parameters = model(X_train, Y_train, X_test, Y_test,num_epochs=150)
# predict(5)
image = np.array(ndimage.imread("/Users/allmight/PycharmProjects/Smash/venv/include/andrew/c1w4/images/bird.jpeg", flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64))
predict(my_image)



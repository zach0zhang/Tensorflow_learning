# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 23:36:44 2018

@author: Zach
"""

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

#配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

#模型保存的路径和文件名
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "model.ckpt"

def train(mnist):
    #定义输入输出placeholder
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name = 'x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name = 'y-input')
    
    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    
    #直接使用mnist_inference.py中定义的前向传播过程
    y = mnist_inference.inference(x, regularizer)

    #定义存储训练轮数的变量
    #这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量(trainable=False)
    #在使用Tensorflow训练神经网络时一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable = False)
    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量(比如global_step)就不需要了
    #tf.trainable_variables返回的就是图上集合GraphKeys.TRAINABLE_VARIABLES中的元素。
    #这个集合的元素就是所有没有指定trainable=False的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    #计算交叉熵作为刻画预测值和真实值之间差距的损失函数。
    #这里使用Tensorflow中提供的sparse_softmax_cross_entropy_with_logits函数来计算交叉熵
    #当分类问题只有一个正确答案时，可以使用这个函数来加速交叉熵的计算
    #MNIST问题的图片中只包含0~9中的一个数字，所以可以使用这个函数来计算交叉熵损失
    #这个函数的第一个参数是神经网络不包括softmax层的前向传播结果，第二个是训练数据的正确答案
    #因为标准答案是一个长度为10的一维数组，使用tf.argmax函数来得到正确答案对应的类别下标
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
    #计算当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, #基础的学习率，随着迭代的进行，更新变量时使用，学习率在这个基础上递减
            global_step,         #当前迭代的轮数
            mnist.train.num_examples / BATCH_SIZE, #过完所有的训练数据需要的迭代次数
            LEARNING_RATE_DECAY) #学习率衰减速度
    #使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
    #这里损失函数包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
    #在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，又要更新每一个参数的滑动平均值。
    #为了一次完成多个参数，Tensorflow提供了tf.control_dependencies和tf.group两种机制
    #也可以这么写 train_op = tf.group(train_step, variables_averages_op)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name = 'train')
    
    #初始化Tensorflow持久化类
    saver = tf.train.Saver()
    #初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        
        for i in range(TRAINING_STEPS):
            #产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            
            #每1000轮保存一次模型
            if i % 1000 == 0:
                #输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。
                #通过损失函数的大小可以大概了解训练的情况
                #在验证数据集上的正确率信息会有一个单独的程序来生成
                print("After %d training step(s), loss on training"
                      "batch is %g." % (step, loss_value))
                #保存当前的模型。注意这里给出了global_strp参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数
                #比如“model.ckpt-1000"表示训练1000轮之后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_DATA", one_hot=True)
    train(mnist)
    
if __name__ == '__main__':
    tf.app.run()
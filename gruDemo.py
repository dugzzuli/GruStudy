import tensorflow as tf

import numpy as np

batch_size = 10  # 批处理大小

input_dim = 100  # 输入维度大小，如单词的词向量维度

output_dim = 128  # 输出神经元数量

inputs = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_dim))

previous_state = tf.random_normal(shape=(batch_size, output_dim))

cell = tf.contrib.rnn.BasicRNNCell(num_units=output_dim)  # 一个BasicRNNCell表示一个时间步

output, state = cell(inputs, previous_state)  # output:输出神经元数量，state:隐藏神经元数量

X = np.ones(shape=(batch_size, input_dim))

print(output.shape)  # (10, 128)

print(state.shape)  # (10, 128)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    o, s = sess.run([output, state], feed_dict={inputs: X})
    print(X)
    print(previous_state.eval())
    print(o)
    print(s)

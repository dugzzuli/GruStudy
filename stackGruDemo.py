# 使用静态堆叠和动态堆叠
# 通过静态生成的RNN网络，生成过程所需的时间会更长，网络所占有的内存会更多，导出的模型会更大，
# 模型中会带有第个序列中间态的信息，利于调试。在使用时必须与训练的样本序列个数相同。
# 通过动态生成的RNN网络，所占用内存较少。模型中只会有最后的状态，在使用时还能支持不同的序列个数。

# 怎么堆叠rnn
# 把多个rnn部件添加到lsit中，通过tf.contrib.rnn.MultiRNNCell函数可以把rnn按顺序链接，
# 堆叠rnn就是多个rnn进行堆叠,每个lstm的单元个数可以不一样。

# gru = tf.contrib.rnn.GRUCell(n_hidden*2)
# lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden)
# mcell = tf.contrib.rnn.MultiRNNCell([lstm_cell,gru])


# 堆叠静态rnn

# stacked_rnn = []
# for i in range(3):
#     stacked_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden))
# mcell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)
# x1 = tf.unstack(x, n_steps, 1)
# outputs, states = tf.contrib.rnn.static_rnn(mcell, x1, dtype=tf.float32)


# 堆叠动态rnn
# stacked_rnn = []
# for i in range(3):
#     stacked_rnn.append(tf.contrib.rnn.LSTMCell(n_hidden))
# mcell = tf.contrib.rnn.MultiRNNCell(stacked_rnn)
# outputs,states  = tf.nn.dynamic_rnn(mcell,x,dtype=tf.float32)#(?, 28, 256)
# outputs = tf.transpose(outputs, [1, 0, 2])
#
# 作者：yanghedada
# 链接：https://www.jianshu.com/p/885e3ab144f4
# 来源：简书
# 简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。




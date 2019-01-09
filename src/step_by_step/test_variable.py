# encoding=utf-8
import tensorflow as tf

# 声明 wl、 w2 两个变盘。这里还通过 seed 参数设定了随机种子， #这样可以保证每次运行得到的结果是一样的。
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))
# 暂时将输入的特征向盘定义为 一个常量。注意这里 x 是一个 lx2 的矩阵。 x = tf.constant([[0 . 7, 0.9)))
# 通过前向传播算法获得神经网络的输出。
x = tf.constant([1.0, 2.0, 3.0])
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
sess = tf.Session()
# 与 3.4.2 中的计算不同，这里不能直接通过 sess . run (y)来获取 y 的取值，
# 因为 wl 和 w2 都还没有运行初始化过程。以下两行分别初始化了 wl 和 w2 两个变量。 sess.run(wl.initializer) #初始化 wl.
sess.run(w2.initializer)  # 初始化 w2。
# 输出[( 3.95757794 ))。
print (sess.run(y))
sess.close()

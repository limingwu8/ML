import tensorflow as tf
from tensorflow.python import debug as tf_debug

def inference():
    x = tf.Variable(0.0, dtype = tf.float32, name='x')

    y = tf.add(tf.add(x**2,tf.multiply(-10.0, x)), 25.0, name='y')

    train = tf.train.GradientDescentOptimizer(0.01).minimize(y)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(init)

    for i in range(500):
        sess.run(train)
        print('x:' + str(sess.run(x)) + ' y:' + str(sess.run(y)))

    sess.close()


if __name__ == "__main__":
    import sys
    print('-------------' + str(sys.argv))
    inference()
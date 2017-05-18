import tensorflow as tf


class PredictModel(object):
    def __init__(self, n_input, n_output, init_scale=0.01):
        self.n_input = n_input
        self.n_output = n_output
        self.init_scale = init_scale

        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.y = tf.placeholder(tf.float32, [None, n_output])
        self.y_ = tf.nn.softmax(tf.add(tf.matmul(self.x, self.weights['w_output']), self.weights['b_output']))

        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.y_, labels=self.y))  # compute costs
        self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.5).minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w_output'] = tf.Variable(tf.random_normal([self.n_input, self.n_output],
                                                               stddev=self.init_scale), dtype=tf.float32)
        all_weights['b_output'] = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32))
        return all_weights

    def partial_fit(self, x=None, y=None):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: x, self.y: y})
        return cost

    def get_result(self, x=None):
        result = self.sess.run(self.y_, feed_dict={self.x: x})
        return result

    def get_weights(self):
        return self.sess.run((self.weights['w_output']))

    def get_biases(self):
        return self.sess.run((self.weights['b_output']))

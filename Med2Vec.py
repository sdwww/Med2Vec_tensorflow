import tensorflow as tf
import numpy as np


class Med2Vec(object):
    def __init__(self, n_input, n_emb, n_hidden, n_demo, n_output, log_eps=1e-8, n_windows=1,
                 optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.5), init_scale=0.01):
        self.n_input = n_input
        self.n_emb = n_emb
        self.n_demo = n_demo
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.log_eps = log_eps
        self.n_windows = n_windows
        self.init_scale = init_scale

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, n_input])
        self.d = tf.placeholder(tf.float32, [None, n_demo])
        self.y = tf.placeholder(tf.float32, [None, n_output])
        self.u = tf.nn.relu(tf.add(tf.matmul(self.x, self.weights['w_emb']), self.weights['b_emb']))
        if n_demo > 0:
            self.v = tf.nn.relu(tf.matmul(tf.concat([self.u, self.d], axis=1),
                                          self.weights['w_hidden']) + self.weights['b_hidden'])
        else:
            self.v = tf.nn.relu(tf.matmul(self.u, self.weights['w_hidden']) + self.weights['b_hidden'])
        self.y_ = tf.nn.softmax(tf.add(tf.matmul(self.v, self.weights['w_output']), self.weights['b_output']))

        # cost
        self.mask = tf.placeholder(tf.float32, [None, 1])
        self.mask1, self.mask2, self.mask3 = self._initialize_mask()
        self.visit_cost = self._initialize_visit_cost()

        self.i_vec = tf.placeholder(tf.float32)
        self.j_vec = tf.placeholder(tf.float32)
        self.emb_cost = self._initialize_visit_cost()

        self.cost = tf.add(self.visit_cost, self.emb_cost)
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w_emb'] = tf.Variable(tf.random_normal([self.n_input, self.n_emb],
                                                            stddev=self.init_scale), dtype=tf.float32)
        all_weights['b_emb'] = tf.Variable(tf.zeros([self.n_emb], dtype=tf.float32))
        all_weights['w_hidden'] = tf.Variable(tf.random_normal([self.n_emb + self.n_demo, self.n_hidden],
                                                               stddev=self.init_scale), dtype=tf.float32)
        all_weights['b_hidden'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w_output'] = tf.Variable(tf.random_normal([self.n_hidden, self.n_output],
                                                               stddev=self.init_scale), dtype=tf.float32)
        all_weights['b_output'] = tf.Variable(tf.zeros([self.n_output], dtype=tf.float32))
        return all_weights

    def _initialize_mask(self):
        mask1 = (self.mask[:-1] * self.mask[1:])[:, None]
        mask2 = (self.mask[:-2] * self.mask[1:-1] * self.mask[2:])[:, None]
        mask3 = (self.mask[:-3] * self.mask[1:-2] * self.mask[2:-1] * self.mask[3:])[:, None]
        return mask1, mask2, mask3

    def _initialize_visit_cost(self):
        if self.n_input > self.n_output:
            t = self.y
        else:
            t = self.x

        forward_results = self.y_[:-1] * self.mask1
        forward_cross_entropy = -(t[1:] * tf.log(forward_results + self.log_eps)
                                  + (1. - t[1:]) * tf.log(1. - forward_results + self.log_eps))

        forward_results2 = self.y_[:-2] * self.mask2
        forward_cross_entropy2 = -(t[2:] * tf.log(forward_results2 + self.log_eps)
                                   + (1. - t[2:]) * tf.log(1. - forward_results2 + self.log_eps))

        forward_results3 = self.y_[:-3] * self.mask3
        forward_cross_entropy3 = -(t[3:] * tf.log(forward_results3 + self.log_eps)
                                   + (1. - t[3:]) * tf.log(1. - forward_results3 + self.log_eps))

        backward_results = self.y_[1:] * self.mask1
        backward_cross_entropy = -(t[:-1] * tf.log(backward_results + self.log_eps)
                                   + (1. - t[:-1]) * tf.log(1. - backward_results + self.log_eps))

        backward_results2 = self.y_[2:] * self.mask2
        backward_cross_entropy2 = -(t[:-2] * tf.log(backward_results2 + self.log_eps)
                                    + (1. - t[:-2]) * tf.log(1. - backward_results2 + self.log_eps))

        backward_results3 = self.y_[3:] * self.mask3
        backward_cross_entropy3 = -(t[:-3] * tf.log(backward_results3 + self.log_eps)
                                    + (1. - t[:-3]) * tf.log(1. - backward_results3 + self.log_eps))

        visit_cost1 = (tf.reduce_sum(forward_cross_entropy) + tf.reduce_sum(backward_cross_entropy)) \
                      / (tf.reduce_sum(self.mask1) + self.log_eps)
        visit_cost2 = (tf.reduce_sum(forward_cross_entropy2) + tf.reduce_sum(backward_cross_entropy2)) \
                      / (tf.reduce_sum(self.mask2) + self.log_eps)
        visit_cost3 = (tf.reduce_sum(forward_cross_entropy3) + tf.reduce_sum(backward_cross_entropy3)) \
                      / (tf.reduce_sum(self.mask3) + self.log_eps)

        visit_cost = visit_cost1
        if self.n_windows == 2:
            visit_cost = visit_cost1 + visit_cost2
        elif self.n_windows == 3:
            visit_cost = visit_cost1 + visit_cost2 + visit_cost3
        return visit_cost

    def _initialize_emb_cost(self):
        w_emb_relu = tf.nn.relu(self.weights['w_emb'])
        norms = tf.reduce_sum(tf.exp(tf.matmul(w_emb_relu, w_emb_relu.T)))
        emb_cost = -tf.log((tf.exp(tf.reduce_sum(w_emb_relu[self.i_vec] * w_emb_relu[self.j_vec], axis=1))
                            / norms[self.i_vec]) + self.log_eps)
        return emb_cost

    def partial_fit(self, x=None, d=None, y=None, mask=None, i_vec=None, j_vec=None):
        if self.n_demo > 0 and self.n_input > self.n_output:
            cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict=
            {self.x: x, self.d: d, self.y: y, self.mask: mask, self.i_vec: i_vec, self.j_vec: j_vec})
        elif self.n_demo > 0 and self.n_input == self.n_output:
            cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict=
            {self.x: x, self.d: d, self.mask: mask, self.i_vec: i_vec, self.j_vec: j_vec})
        elif self.n_demo == 0 and self.n_input > self.n_output:
            cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict=
            {self.x: x, self.y: y, self.mask: mask, self.i_vec: i_vec, self.j_vec: j_vec})
        else:
            cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict=
            {self.x: x, self.mask: mask, self.i_vec: i_vec, self.j_vec: j_vec})
        return cost

    def get_code_representation(self, x=None, d=None):
        if self.n_demo > 0 and self.n_input > self.n_output:
            code_representation = self.sess.run(self.u, feed_dict={self.x: x, self.d: d})
        else:
            code_representation = self.sess.run(self.u, feed_dict={self.x: x})
        return code_representation

    def get_visit_representation(self, x=None, d=None):
        if self.n_demo > 0 and self.n_input > self.n_output:
            visit_representation = self.sess.run(self.v, feed_dict={self.x: x, self.d: d})
        else:
            visit_representation = self.sess.run(self.v, feed_dict={self.x: x})
        return visit_representation

    def get_weights(self):
        return self.sess.run((self.weights['w_emb'], self.weights['w_hidden'], self.weights['w_output']))

    def get_biases(self):
        return self.sess.run((self.weights['b_emb'], self.weights['b_hidden'], self.weights['b_output']))
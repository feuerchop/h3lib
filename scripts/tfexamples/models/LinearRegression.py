'''
Trivial
'''

import numpy as np
import tensorflow as tf


class LinearRegression(object):
   '''
   An implementation using tensorflow
   '''

   def __init__(self, feature_size=2, learning_rate=0.01, max_epochs=1000, display_epochs=100,
                summary_output='summaries/'):
      '''
      Constructor:
      build the graph
      '''

      self.learning_rate = tf.constant(learning_rate, dtype=tf.float32)
      self.max_epochs = tf.constant(max_epochs, dtype=tf.int8)
      self.display_epochs = tf.constant(display_epochs, dtype=tf.int8)
      self.feature_size = feature_size
      self.input_X = tf.placeholder(dtype=tf.float32, shape=(None, feature_size))
      self.input_Y = tf.placeholder(dtype=tf.float32)
      self.weight = tf.Variable(np.random.randn(feature_size, 1), dtype=tf.float32, name='W')
      self.bias = tf.Variable(np.random.randn(), dtype=tf.float32, name='b')
      self.pred = tf.add(tf.matmul(self.input_X, self.weight), self.bias)
      self.loss = tf.reduce_mean(tf.pow(self.pred - self.input_Y, 2))
      tf.scalar_summary('loss', self.loss)
      self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
      self.global_step = tf.Variable(0, dtype=tf.int8, trainable=False)
      self.summary_writer = tf.train.SummaryWriter(summary_output)

   def train_step(self):
      # one training step
      grads = self.optimizer.compute_gradients(loss=self.loss, var_list=[self.weight, self.bias])
      return self.optimizer.apply_gradients(grads_and_vars=grads, global_step=self.global_step, name='apply_grad')

   def train(self, X, Y):
      init = tf.initialize_all_variables()
      merge_summaries = tf.merge_all_summaries()
      n_dim = X.shape[1]
      with tf.Session() as sess:
         sess.run(init)
         for e in np.arange(100):
            # each epoch
            step = self.train_step()
            _, loss, summaries = sess.run([step, self.loss, merge_summaries],
                                          feed_dict={self.input_X: X, self.input_Y: Y})
            print 'epoch ', e, ': total loss: ', loss, 'W: ', sess.run(self.weight), 'b: ', sess.run(self.bias)
            self.summary_writer.add_summary(summaries, global_step=e)


if __name__ == '__main__':
   from sklearn.datasets import make_regression

   n_dim = 2
   X, y = make_regression(100, n_features=n_dim, n_informative=1, n_targets=1)
   clf = LinearRegression(feature_size=n_dim, max_epochs=10, display_epochs=150)
   clf.train(X, y)

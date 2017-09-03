'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 20000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
   # Conv2D wrapper, with bias and relu activation
   x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
   x = tf.nn.bias_add(x, b)
   return tf.nn.relu(x)


def maxpool2d(x, k=2):
   # MaxPool2D wrapper
   return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                         padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
   # Reshape input picture
   x = tf.reshape(x, shape=[-1, 28, 28, 1])

   # Convolution Layer
   conv1 = conv2d(x, weights['wc1'], biases['bc1'])
   # Max Pooling (down-sampling)
   conv1 = maxpool2d(conv1, k=2)

   # Convolution Layer
   conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
   # Max Pooling (down-sampling)
   conv2 = maxpool2d(conv2, k=2)

   # Fully connected layer
   # Reshape conv2 output to fit fully connected layer input
   fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
   fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
   fc1 = tf.nn.relu(fc1)
   # Apply Dropout
   fc1 = tf.nn.dropout(fc1, dropout)

   # Output, class prediction
   out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
   return out, conv1, conv2


# Store layers weight & bias
weights = {
   # 5x5 conv, 1 input, 32 outputs
   'wc1': tf.Variable(tf.random_normal([5, 5, 1, 3])),
   # 5x5 conv, 32 inputs, 64 outputs
   'wc2': tf.Variable(tf.random_normal([5, 5, 3, 32])),
   # fully connected, 7*7*64 inputs, 1024 outputs
   'wd1': tf.Variable(tf.random_normal([7 * 7 * 32, 64])),
   # 1024 inputs, 10 outputs (class prediction)
   'out': tf.Variable(tf.random_normal([64, n_classes]))
}

biases = {
   'bc1': tf.Variable(tf.random_normal([3])),
   'bc2': tf.Variable(tf.random_normal([32])),
   'bd1': tf.Variable(tf.random_normal([64])),
   'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred, conv1, conv2 = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# summaries
tf.scalar_summary('training loss', cost)
tf.scalar_summary('training accuracy', accuracy)
# tf.image_summary('conv1', conv1)
# tf.image_summary('conv2', conv2[:,:,:,:3])
summary_writer = tf.train.SummaryWriter('summaries/convnet-example/', graph=tf.get_default_graph())

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
   sess.run(init)
   step = 1
   # Keep training until reach max iterations
   while step * batch_size < training_iters:
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      if step % display_step == 9:
         # record summaries every 100 steps
         run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
         run_metadata = tf.RunMetadata()
         merge_summaries = tf.merge_all_summaries()
         # Calculate batch loss and accuracy
         _, loss, acc, summaries = sess.run([optimizer, cost, accuracy, merge_summaries],
                                            feed_dict={x: batch_x,
                                                       y: batch_y,
                                                       keep_prob: 1.},
                                            options=run_options, run_metadata=run_metadata)
         print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
               "{:.6f}".format(loss) + ", Training Accuracy= " + \
               "{:.5f}".format(acc))
         summary_writer.add_run_metadata(run_metadata, 'step%d'%step, step)
         summary_writer.add_summary(summaries, step)
      else:
         sess.run(optimizer, feed_dict={x: batch_x,
                                        y: batch_y,
                                        keep_prob: dropout})

      step += 1

   print("Optimization Finished!")

   # Calculate accuracy for 256 mnist test images
   print("Testing Accuracy:", \
         sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                       y: mnist.test.labels[:256],
                                       keep_prob: 1.}))

   # let us check the conv layers
   with tf.name_scope('test_part'):
      # Applying encode and decode over test set
      c1, c2 = sess.run(
         [conv1, conv2], feed_dict={x: mnist.test.images[:10]})
      # Compare original images with their reconstructions
      f, a = plt.subplots(4, 10, figsize=(10, 4))
      for i in range(10):
         a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
         a[1][i].imshow(np.reshape(c1[i, :, :, 0], (14, 14)))
         a[2][i].imshow(np.reshape(c1[i, :, :, 1], (14, 14)))
         a[3][i].imshow(np.reshape(c1[i, :, :, 2], (14, 14)))
      f.show()
      plt.draw()
      plt.waitforbuttonpress()

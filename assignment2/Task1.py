import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random
import warnings
from sklearn.neighbors import KNeighborsClassifier


# mnist = tf.keras.datasets.mnist
# # print(mnist)


def conv2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# initialize weight
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# initialize bias
def bias_variable(shape):
    return tf.Variable(tf.zeros(shape))


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("classifier 1, KNN starts")
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(mnist.train.images, mnist.train.labels)

score = kn.score(mnist.test.images, mnist.test.labels)

print(" KNN, score: {:.6f}".format(score))

print("classifier 1, Convolutional neural network starts")

input = tf.placeholder(tf.float32, [None, 784])
input_image = tf.reshape(input, [-1, 28, 28, 1])

y = tf.placeholder(tf.float32, [None, 10])

# convolutional layer 1
W1 = weight_variable([5, 5, 1, 32])
b1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(input_image, W1) + b1)
# pooling layer 1
h_pool1 = max_pool(h_conv1)

# convolutional layer 2
W2 = weight_variable([5, 5, 32, 64])
b2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
# pooling layer 2
h_pool2 = max_pool(h_conv2)

# fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout, prevent overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))

# training
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch = mnist.train.next_batch(50)
        if i % 1000 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                input: batch[0], y: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={input: batch[0], y: batch[1], keep_prob: 0.5})

    print('Convolutional neural network, test accuracy %g' % accuracy.eval(
        feed_dict={input: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

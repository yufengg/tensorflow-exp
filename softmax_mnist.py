import tensorflow as tf

# read data in

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create and run the MNIST data
# The softmax will be run on the model of
# y = W*x + b
# Where all values are tensors (matrices in this case)


# ########################
# Step 1: Set up the model
# ########################

# Create a 2D tensor of arbitrary width, and 784 length.
# This is to give space for the computations for
# an arbitrary number of 28x28 images
x = tf.placeholder(tf.float32, [None, 784])

# Variables for the weights and biases. 
# Variables are modifiable tensors that live in TF's
# graph of interacting operations.
# These tend to be ML model parameters.
# We have 10 here because we are recognizing values 0-9
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# tf.matmul() is matrix multiply, we do x,W to match matrix dims
y= tf.nn.softmax(tf.matmul(x,W) + b)


# ########################
# Step 2: Train the model
# ########################

# Use cross-entropy for the cost func
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# use gradient descent to train, learning rate of 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# For each step(aka loop), we get a 'batch' of 100 images from the training set.
# Then use the train_step to compute values to replace the placeholders.
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# ########################
# Step 3: Evaluate the model
# ########################

# true/false for model accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# average it over all values
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))





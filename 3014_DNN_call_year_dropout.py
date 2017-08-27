import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

#input data
test_data = np.loadtxt('3014_testset.csv', delimiter=',', dtype=np.float32)
x_test = test_data[:,0:-1]
y_test = test_data[:,[-1]]
keep_prob = tf.placeholder(tf.float32)


X = tf.placeholder(tf.float32, shape=[None,40])
Y = tf.placeholder(tf.float32, shape=[None,1])

#nural network layers
W1 = tf.get_variable('W1', shape=[40,50],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([50]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable('W2', shape=[50,50],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([50]))
L2 = tf.matmul(L1, W2) + b2
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable('W3', shape=[50,50],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([50]))
L3 = tf.matmul(L2, W3) + b3
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable('W4', shape=[50,20],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([20]))
L4 = tf.matmul(L3, W4) + b4
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable('W5', shape=[20,1],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(L4, W5) + b5

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# call Model
saver = tf.train.Saver()
save_path = saver.restore(sess, 'model_year')

Prediction = sess.run(hypothesis,feed_dict={X:x_test, keep_prob:1})
print('\nPrediction:',Prediction, '\nDelay time:',y_test)

#accuracy
Accuracy = (Prediction - y_test)
print('\nAccuracy:',Accuracy)

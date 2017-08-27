import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('3014.csv', delimiter=',', dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

test_data = np.loadtxt('3014_testset.csv', delimiter=',', dtype=np.float32)
x_test = test_data[:,0:-1]
y_test = test_data[:,[-1]]

#print(x_data.shape, x_data, len(x_data))
#print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape=[None,40])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([40,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

opyimizer = tf.train.GradientDescentOptimizer(learning_rate=9e-6)
train = opyimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(505000):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
    feed_dict={X: x_data, Y: y_data})
    if step % 5000 == 0:
        print(step, 'Cost: ', cost_val,)

Prediction = sess.run(hypothesis,feed_dict={X:x_test})
#print('\nPrediction:',Prediction, '\nDelay time:',y_test)

#accuracy
Accuracy = (Prediction/y_test)*100
print('\nAccuracy:',Accuracy)

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

xy = np.loadtxt('./data/3014_training.csv', delimiter=',', dtype=np.float32)
#x_data = xy[0:38,0:-1]
#y_data = xy[0:38,[-1]]

#print(x_data.shape, x_data, len(x_data))
#print(y_data.shape, y_data)

#set pramater
learning_rate = 1e-6
training_epoch = 500000
batch_size = 100
data_size = 270

X = tf.placeholder(tf.float32, shape=[None,40])
Y = tf.placeholder(tf.float32, shape=[None,1])

#nural network layers
W1 = tf.get_variable('W1', shape=[40,50],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([50]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable('W2', shape=[50,30],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([30]))
L2 = tf.matmul(L1, W2) + b2

W3 = tf.get_variable('W3', shape=[30,20],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([20]))
L3 = tf.matmul(L2, W3) + b3

W4 = tf.get_variable('W4', shape=[20,1],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(L3, W4) + b4

# cost, optimizer
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#training
#for step in range(training_range):
#    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
#    feed_dict={X: x_data, Y: y_data})
#    if step % 10000 == 0:
#        print(step, 'Cost: ', cost_val)

for epoch in range(training_epoch):
    avg_cost = 0
    total_batch = int(data_size/batch_size)

#    for i in range(total_batch):
#        batch_xs, batch_ys = xy.next_batch(batch_size)
#        c,_ = sess.run([cost, train], feed_dict={X:batch_xs, Y:batch_ys})
#        avg_cost += c / total_batch
    i = 0
    for j in range(int(data_size/batch_size+1)):
        k = i
        i += batch_size
        x_data = xy[k:i, 0:-1]
        y_data = xy[k:i, [-1]]
        c,_ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
        avg_cost += c / total_batch

    if epoch % 5000 == 0:
        print('Epoch: ', '%04d'%(epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print("Learning finished")

#Save model
saver = tf.train.Saver()
save_path = saver.save(sess, 'model1_testset')
#print('Model saved in file:', save_path)

#Evaluation model
test_data = np.loadtxt('./data/3014_2017.csv', delimiter=',', dtype=np.float32)
x_test = test_data[:,0:-1]
y_test = test_data[:,[-1]]
#print(x_test)

Prediction = sess.run(hypothesis,feed_dict={X:x_test})
#print('\nPrediction:',Prediction, '\nDelay time:',y_test)

#accuracy
Accuracy1 = (Prediction/y_test)*100
print('\nAccuracy1:',Accuracy1)

Accuracy2 = Prediction - y_test
print('\nAccuracy2:',Accuracy2)

#print result
x = Accuracy1.astype(np.int64)
y = Accuracy2.astype(np.int64)

with open('./Data/result.txt','a') as f:
   f.write("\nLeraning rate : %f\nEpoch : %d\nBatch size : %d\n\
Data size : %d\n" %(learning_rate, training_epoch, batch_size, data_size))
   f.write('\nPrediction = ')
   f.write('\n%s'%str(x))
   f.write('\n\nTarget = ')
   f.write('\n%s'%str(y))
   f.write('\n------------------------------------------------------\
-------------------------\n\n')

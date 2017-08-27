import tensorflow as tf
import numpy as np
import datetime

time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

tf.set_random_seed(777)

xy = np.loadtxt('./data/3014_dataset.csv', delimiter=',', dtype=np.float32)
test_data = np.loadtxt('./data/3014_testset.csv', delimiter=',', dtype=np.float32)

#Evaluation model
x_testset = test_data[:,0:-1]
y_testset = test_data[:,[-1]]
x_dataset = xy[:,0:-1]
y_dataset = xy[:,[-1]]

#print(x_data.shape, x_data, len(x_data))
#print(y_data.shape, y_data)

#set pramater
learning_rate = 1e-5
training_epoch = 6000
batch_size = 100
data_size = 335
keep_prob_number = 1

X = tf.placeholder(tf.float32, shape=[None,40])
Y = tf.placeholder(tf.float32, shape=[None,1])
keep_prob = tf.placeholder(tf.float32)

#nural network layers
W1 = tf.get_variable('W1', shape=[40,50],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([50]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable('W2', shape=[50,50],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([50]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable('W3', shape=[50,50],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([50]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable('W4', shape=[50,20],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([20]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable('W5', shape=[20,1],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(L4, W5) + b5

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
rmse_dataset = []
rmse_testset = []
count_epoch = []

with open('./Data/result.txt','a') as f:
    f.write("\n\n----------------------------------------------------")
    f.write(time_now)
    f.write("\nLeraning rate : %f\nEpoch : %d\nBatch size : %d\n\
#Data size : %d\nKeep prob : %d" %(learning_rate, training_epoch, batch_size,
data_size, keep_prob_number))

    f.write("\n\nlearning start")

#learning
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
        c,_ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data, keep_prob:keep_prob_number})
        avg_cost += c / total_batch

    if epoch % 10 == 0:
        prediction_dataset = sess.run(hypothesis,feed_dict={X:x_dataset, keep_prob:1})
        prediction_testset = sess.run(hypothesis,feed_dict={X:x_testset, keep_prob:1})
        rmse_data = int(np.sqrt(np.mean(np.square(y_dataset - prediction_dataset))))
        rmse_test = int(np.sqrt(np.mean(np.square(y_testset - prediction_testset))))

        rmse_dataset.append(rmse_data)
        rmse_testset.append(rmse_test)
        count_epoch.append(epoch)

#print result
    if epoch % 1000 == 0:
        print('Epoch:', '%04d'%(epoch + 1), 'Cost:', '{:.9f}'.format(avg_cost),
        'DataError:', rmse_data, 'TestError:', rmse_test)

        with open('./Data/result.txt','a') as f:
           f.write("\nEpoch: %d  Cost: %f  DataError: %d  TestError: %d"
           %(epoch + 1, avg_cost, rmse_data, rmse_test))

print("Learning finished")

#Save model
saver = tf.train.Saver()
save_path = saver.save(sess, './save/3014_noepoch')
#print('Model saved in file:', save_path)

#print result
x = prediction_testset.astype(np.int64)
x = x.reshape(1,67)
y = y_testset.reshape(1,67)

with open('./Data/result.txt','a') as f:
   f.write('\n\ntestdata = ')
   f.write('\n%s'%str(y))
   f.write('\n\nprediction = ')
   f.write('\n%s'%str(x))
   f.write('\n\ntestdata - prediction = ')
   f.write('\n%s'%str(y-x))

   #plot graph
   import matplotlib
   matplotlib.use('TkAgg')
   import matplotlib.pyplot as plt
   plt.plot(count_epoch, rmse_dataset, 'g-')
   plt.plot(count_epoch, rmse_testset, 'r-')
   plt.show()

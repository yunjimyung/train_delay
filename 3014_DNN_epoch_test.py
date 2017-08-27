import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

xy = np.loadtxt('3014.csv', delimiter=',', dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

#print(x_data.shape, x_data, len(x_data))
#print(y_data.shape, y_data)

#set pramater
learning_rate = 7e-6
training_epoch = 15
batch_size = 10
data_size = 37

i = 0
for j in range(int(data_size/batch_size+1)):
    k = i
    i += batch_size
    x_data = xy[k:i, 0:-1]
    y_data = xy[k:i, [-1]]
    print(x_data.shape,y_data.shape,k,i)
    print(x_data,y_data)


'''for epoch in range(training_epoch):
    avg_cost = 0
    total_batch = int(data_size/batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = xy.next_batch(batch_size)
        c,_ = sess.run([cost, train], feed_dict={X:batch_xs, Y:batch_ys})
        avg_cost += c / total_batch

    print('Epoch: ', '%04d'%(epoh + 1), 'cost =', '{:.9f}'.format(avg_cost))

print("Learning finished")'''

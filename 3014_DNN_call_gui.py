import tensorflow as tf
import numpy as np
from tkinter import *
import datetime

tf.set_random_seed(777)

#input data
test_data = np.loadtxt('3014_6.csv', delimiter=',', dtype=np.float32)
x_test = test_data[:,0:-1]
y_test = test_data[:,[-1]]


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

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# call Model
saver = tf.train.Saver()
save_path = saver.restore(sess, 'model1.ckpt')

Prediction = sess.run(hypothesis,feed_dict={X:x_test})
#print('\nPrediction:', Prediction, '\nDelay time:',y_test)

#convert_scalar
#convert_scalar = Prediction[0]
#a = '%.2f'%convert_scalar  #소수 두째자리까지 표시
Hour = int(Prediction[1]//60)#60으로 나누고 몪을 표시
Minute = int(Prediction[1]%60) #60으로 나누고 나머지를 정수로 표시
#Hour_y = int(y_test[1]//60)
#Minute_y  = int(y_test[1]%60)

prediction_delaytime = Hour,'시간',Minute,'분'
#print('실제 지연시간:',Hour_y,'시간',Minute_y,'분')

#accuracy
#Accuracy = (Prediction[1]/y_test[1])*100
#print('정확도 :',int(Accuracy),'%')

#=================== GUI =================================================

root = Tk()
root.title('JENNY')
root.geometry('300x350+100+100')
time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')



class GuiForm:

    def __init__(self):

        self.F1 = Frame(root, width = 300, height = 50)
        self.F1.grid(row=0,column=0)

        self.F2 = Frame(root, width = 300, height =300)
        self.F2.grid(row=1,column=0)

        self.F3 = Frame(root, width = 300, height =100)
        self.F3.grid(row=2,column=0)

        self.Title1 = Label(self.F1, font=('arial',22,'bold'), text=' Korail 열차지연 예측 프로그램',
                           fg='navy', bd=10, anchor='w')
        self.Title1.grid(row=0,column=0)

        self.Title2 = Label(self.F1, font=('arial',20,'bold'), text=time_now,
                           fg='blue', bd=10, anchor='w')
        self.Title2.grid(row=1,column=0)


        self.Blank1 = Label(self.F2,font=('arial',10,'bold'),text=' ')
        self.Blank1.grid(row=0,column=0)

        self.Text1 = Label(self.F2, font=('arial',22,'bold'), text=' 화물열차 열번 :',
                          fg='black', bd=10, anchor='w')
        self.Text1.grid(row=1,column=0)

        self.Text1_1 = Label(self.F2, font=('arial',30,'bold'), text=train_number,
                            fg='black', bd=10, anchor='w')
        self.Text1_1.grid(row=1,column=1)


        self.Text2 = Label(self.F2, font=('arial',22,'bold'), text=' 예측 지연시간 :',
                          fg='Red', bd=10, anchor='w')
        self.Text2.grid(row=2,column=0)

        self.Text2_1 = Label(self.F2, font=('arial',25,'bold'), text='-------',
                            fg='Red', bd=10, anchor='w')
        self.Text2_1.grid(row=2,column=1)


        self.Text3 = Label(self.F2, font=('arial',22,'bold'), text=' 도착 예정시간 :',
                           fg='blue', bd=10, anchor='w')
        self.Text3.grid(row=3,column=0)

        self.Text3_1 = Label(self.F2, font=('arial',25,'bold'), text='-------',
                            fg='blue', bd=10, anchor='w')
        self.Text3_1.grid(row=3,column=1)

        def btn1Click():
            self.Text2_1['text'] = prediction_delaytime
            self.Text3_1['text'] = time_of_arrival

        self.Blank2 = Label(self.F2,font=('arial',20,'bold'),text='   ')
        self.Blank2.grid(row=4,column=0)

        self.Button1=Button(self.F3,padx=30,pady=3,bd=5,fg='black',font=('arial',15,'bold'),
                 text='예측', bg="red", command=btn1Click)
        self.Button1.pack(side=LEFT, padx=15, pady=0)

        self.Button2=Button(self.F3,padx=30,pady=3,bd=5,fg='black',font=('arial',15,'bold'),
                 text='종료',bg="green", command=root.quit)
        self.Button2.pack(padx=15, pady=0)

def calculate_dalaytime(xrois_endtime):
    delay_hour = int(xrois_endtime[0:2])+Hour
    delay_minute = int(xrois_endtime[3:5])+Minute
    if delay_minute >= 60:
        delay_hour = delay_hour + delay_minute//60
        delay_minute = delay_minute%60
        if delay_hour > 24:
            delay_hour = delay_hour - 24
    else:
        if delay_hour > 24:
            delay_hour = delay_hour - 24

    result = str(delay_hour)+' 시 '+ str(delay_minute)+' 분'
    return result

train_number = '3014'
xrois_endtime = '23:30:00'
time_of_arrival = calculate_dalaytime(xrois_endtime)

Program_start = GuiForm()
root.mainloop()

import pandas as pd
import numpy as np
from sklearn import model_selection
import tensorflow as tf

df=pd.read_csv('Fdata.csv')
df.drop('id',axis=1,inplace=True)
df=pd.get_dummies(df,columns=['label'])
#print(df)
x=df.ix[:,(0,1,2,3,4,5,6,7,8)].values
y=df.ix[:,(9,10)].values
#print(x,y)

x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.10,random_state=100)

path=('C:\\Users\\Raghuram\\Desktop\\github\\bcancer\\saver\\save_net.ckpt')
n_nodes_hl1=60
n_nodes_hl2=50
n_nodes_hl3=50
n_nodes_hl4=50
training_epochs=1000
display_step=100

n_input=x_train.shape[1]
n_classes=y_train.shape[1]

x=tf.placeholder('float',[None,n_input])
y=tf.placeholder('float',[None,n_classes])

def multilayerperceptron(x):
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([n_input,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl4])),'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}
    output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl4,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1=tf.add(tf.matmul(x,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1=tf.nn.relu(l1)

    l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2=tf.nn.relu(l2)

    l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3=tf.nn.relu(l3)

    l4=tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l4=tf.nn.relu(l4)

    out=tf.add(tf.matmul(l4,output_layer['weights']),output_layer['biases'])
    return out

predictions=multilayerperceptron(x)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
#loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(predictions), axis=0))
with tf.Session() as sess:
    saver=tf.train.Saver()    
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,path)
    for epoch in range(training_epochs):
        _,c=sess.run([optimizer,cost],feed_dict={x:x_train,y:y_train})
        if(epoch%display_step==0):
            print("Epoch:",'%04d'%(epoch),"cost=","{:.9f}".format(c))
    print('Optimization Finished!!!')
    correct_prediction=tf.equal(tf.argmax(predictions,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
    print('Accuracy:',accuracy.eval({x:x_test,y:y_test}))
    sp=saver.save(sess,save_path=path)
    print('Madel saved at:',sp)
    sess.close()
    

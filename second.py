import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler,MinMaxScaler
import tensorflow as tf

train = pd.read_csv('train.csv')

x_train = train.iloc[:,2:-1]
y_train = train.SalePrice.values
# print x_train


# print dataset.iloc[:,-1]
# print y_train.shape

for i in range(0,x_train.shape[1]):
	if 'object' == x_train.iloc[:,i].dtype:
		x_train.iloc[:,i] = LabelEncoder().fit_transform(x_train.iloc[:,i].fillna('0'))






test = pd.read_csv('test.csv')
x_test = test.iloc[:,2:]

#print x_test
for i in range(0,x_test.shape[1]):
	if 'object' == x_test.iloc[:,i].dtype:
		x_test.iloc[:,i] = LabelEncoder().fit_transform(x_test.iloc[:,i].fillna('0'))

print (x_train)
print( x_test)

sc = MinMaxScaler()
x_test = sc.fit_transform(x_test)
x_train = sc.fit_transform(x_train)


# x_train = x_train.values
# x_test = x_test.values


print( x_train)
print( x_test)

def neural_net(X_data, input_dim):
	W_1 = tf.Variable(tf.random_uniform([input_dim,30]))
	B_1 = tf.Variable(tf.zeros([30]))
	layer_1 = tf.add(tf.matmul(X_data,W_1),B_1)
	layer_1 = tf.nn.relu(layer_1)


	W_2 = tf.Variable(tf.random_uniform([30,20]))
	b_2 = tf.Variable(tf.zeros([20]))
	layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
	layer_2 = tf.nn.relu(layer_2)

	W_3 = tf.Variable(tf.random_uniform([20,10]))
	b_3 = tf.Variable(tf.zeros([10]))
	layer_3 = tf.add(tf.matmul(layer_2,W_3), b_3)
	layer_3 = tf.nn.relu(layer_3)


	# layer 2 multiplying and adding bias then activation function
	W_O = tf.Variable(tf.random_uniform([10,1]))
	b_O = tf.Variable(tf.zeros([1]))
	output = tf.add(tf.matmul(layer_3,W_O), b_O)

	return output

xs = tf.placeholder('float')
ys = tf.placeholder('float')

output = neural_net(xs,78)
cost = tf.reduce_mean(tf.square(output - ys))

train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	for i in range(100):
		for j in range(x_train.shape[0]):
			sess.run([cost,train],feed_dict={xs:x_train[j,:].reshape(1,x_train.shape[1]),ys:y_train[j]})

		print('Epoch :',i,'Cost :',sess.run(cost, feed_dict={xs:x_train,ys:y_train}))


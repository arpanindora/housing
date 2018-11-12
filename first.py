import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv('train.csv')

x_train = dataset.iloc[:,2:-1]
y_train = dataset.iloc[:,-1].values
# print x_train


print dataset.iloc[:,-1]
print y_train.shape

for i in range(0,x_train.shape[1]):
	if 'object' == x_train.iloc[:,i].dtype:
		x_train.iloc[:,i] = LabelEncoder().fit_transform(x_train.iloc[:,i])






dataset = pd.read_csv('test.csv')
x_test = dataset.iloc[:,2:]

#print x_test
for i in range(0,x_test.shape[1]):
	if 'object' == x_test.iloc[:,i].dtype:
		x_test.iloc[:,i] = LabelEncoder().fit_transform(x_test.iloc[:,i])
"""
print x_train
print x_test
"""
x_train = x_train.values


x_test = x_test.values


"""
print x_train
print x_test
"""
sc = StandardScaler()

x_test = sc.fit_transform(x_test)
x_train = sc.fit_transform(x_train)


# print x_train
print x_test

print x_train.shape
print y_train

y_train= y_train.reshape(1460,1)
print y_train

# y = y_train.reshape(1,1460)
net = Sequential()

net.add(Dense(50,input_dim = 78,activation='relu'))
net.add(Dense(30,activation='relu'))
net.add(Dense(10,activation='relu'))
net.add(Dense(1, activation= 'linear'))

net.compile(optimizer='adam',loss='mse')
net.fit(x_train,y_train,batch_size=50,epochs=10)

y_pred = net.predict(x_train)

print y_pred.shape,y_train.shape
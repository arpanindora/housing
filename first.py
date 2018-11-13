import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler,MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import adam
from IPython.display import clear_output
import matplotlib.pyplot as plt
import matplotlib.animation as animation


plt.ion()
class PlotLosses(keras.callbacks.Callback):
    # def __init__(self):
    #     self.i = 0
    #     self.x = []
    #     self.losses = []
    #     self.val_losses = []
        
    #     self.fig = plt.figure()
        

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.ax1.set_title(label='loss')
        self.ax2.set_title(label='validation loss')
        
        # self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        # self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        if epoch%100==0:
        # clear_output(wait=False)
	        self.ax1.plot(self.x, self.losses)
	        self.ax2.plot(self.x, self.val_losses)
	        # plt.legend()
	        plt.pause(0.000001)
        
plot_losses = PlotLosses()
plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)


# ani = animation.FuncAnimation(fig, plot_losses.on_epoch_end,fargs=(plot_losses.x,plot_losses.losses), interval=100)
# plt.show()

train = pd.read_csv('train.csv')

x_train = train.iloc[:,2:-1]
y_train = train.SalePrice
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
"""
print x_train
print x_test
"""


x_test[x_test==np.inf]=np.nan
x_test.fillna(x_test.mean(), inplace=True)
x_train[x_train==np.inf]=np.nan
x_train.fillna(x_train.mean(), inplace=True)


sc = StandardScaler()
# sc = MinMaxScaler()

x_test = sc.fit_transform(x_test)
x_train = sc.fit_transform(x_train)

# x_train = x_train.values
# x_test = x_test.values


# print( x_train)
# print( x_test)




# print x_train
# print x_test

# print x_train.shape
# print y_train


# y = y_train.reshape(1460,1)
# print y_train.shape


# y = y_train.reshape(1460,1)
net = Sequential()

net.add(Dense(34,input_dim = 78,activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
# net.add(Dense(16,activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
# net.add(Dropout(0.4))
net.add(Dense(17,activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
net.add(Dropout(0.5))
net.add(Dense(4,activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))

# net.add(Dense(2,activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
# net.add(Dropout(0.1))
net.add(Dense(1, activation= 'linear', kernel_initializer='random_uniform', bias_initializer='zeros'))

# adam = keras.optimizers.Adam(lr=0.01,decay=0.1)
epochs = 5000
learning_rate = 0.1
decay_rate = learning_rate / epochs
# momentum = 0.8
adam = adam(lr=learning_rate, decay=decay_rate)

net.compile(optimizer=adam,loss='mse')


history = net.fit(x_train,y_train,batch_size=200,epochs=epochs,callbacks=[plot_losses])

y_pred = net.predict(x_test)

# print (y_pred.shape,y_train.shape)
# print(test.Id.shape)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred[:,0]})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# print(history.history.keys())

# print(history.history.val_loss)

# print(history.history.loss)

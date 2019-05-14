"""S Suprabath Reddy
EE15BTECH11026
"""

"""Python 3.5"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import combinations
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import keras
from keras.models import Sequential
from sklearn import metrics
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

energy = pd.read_csv("IIT_A_Emergency_Panel_11_4.csv")
energy1 = pd.read_csv("IIT_A_Emergency_Panel_11_5.csv")

f = open('result.csv', 'w')
f.write("Machine Learning Technique,Mean Absolute Error\n")

histograms = energy.hist(figsize=(16, 16), bins=20)
plt.show()

energy['average_voltage'].describe()
energy['average_current'].describe()
energy['total_active_power'].describe()

plt.plot(np.arange(len(energy)),energy['average_voltage'],color = 'blue',linewidth=1, linestyle="-")
plt.show()
plt.plot(np.arange(len(energy)),energy['average_current'],color = 'blue',linewidth=1, linestyle="-")
plt.show()
plt.plot(np.arange(len(energy)),energy['total_active_power'],color = 'blue',linewidth=1, linestyle="-")
plt.show()

plt.plot(np.arange(len(energy)),energy['total_power_factor'],color = 'blue',linewidth=1, linestyle="-")
plt.show()

plt.plot(np.arange(len(energy)),energy['total_active_power'],color = 'red',linewidth=1, linestyle="-")
plt.plot(np.arange(len(energy1)),energy1['total_active_power'],color = 'green',linewidth=1, linestyle="-")
plt.show()


energy['date'] = pd.to_datetime(energy['date'])
energy['just_date'] = energy['date'].dt.date

plt.plot(energy['just_date'],energy["total_active_power"],'ro')
plt.show()

X_train = energy.drop(["cumulative_energy_KWh",'frequency','total_active_power','date','just_date'], axis=1)
y_train = energy["total_active_power"]

X_test = energy1.drop(["cumulative_energy_KWh",'frequency','total_active_power','date'], axis=1)
y_test = energy1["total_active_power"]



#Implementing Support Vector Regression

from sklearn import svm

clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)

error_svr = np.sqrt(metrics.mean_absolute_error(y_test,y_predict))
print (error_svr)
f.write("Support Vector Regression" + ',' + str(error_svr) + '\n')


#Implementing ANN

model = Sequential()

# The Input Layer :
model.add(Dense(X_train.shape[1], kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
model.add(Dense(64, kernel_initializer='normal',activation='relu'))
model.add(Dense(1024, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()

model.fit(X_train,y_train,validation_split=0.1,batch_size=1024,epochs=500)

y_predict = model.predict(X_test)
error_neural = np.sqrt(metrics.mean_absolute_error(y_test,y_predict))
print (error_neural)

f.write("Artificial Neural Network" + ',' + str(error_neural) + '\n')

# Implementing LSTM Network

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()

model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(units=256))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=10, batch_size=1024, verbose=2)

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

y_predict = model.predict(X_test)
error_lstm = np.sqrt(metrics.mean_absolute_error(y_test,y_predict))

f.write("Long Short Term Memory Network" + ',' + str(error_lstm) + '\n')

plt.title("Predited Data Vs Test Data")
#X_train.plot.line(X_train['just_date'], y_train,linewidth=5, linestyle="-")
plt.plot(np.arange(len(y_predict)),y_predict,color = 'blue',linewidth=1, linestyle="-")
plt.plot(np.arange(len(y_predict)),y_test,color = 'green',linewidth=1, linestyle="-")
plt.show()











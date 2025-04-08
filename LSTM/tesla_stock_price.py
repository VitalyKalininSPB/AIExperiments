import pandas as pd
import pandas_datareader as pdr

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


df = pdr.get_data_tiingo('TSLA', api_key='XXXXX')
#df.to_csv('TSLA.csv')
#df = pd.read_csv('TSLA.csv')
df.head(12)
df.tail(3)

dataframeNew = df.reset_index()['close']
dataframeNew.shape

plt.plot(dataframeNew)

scaler = StandardScaler()

dataframeNew = scaler.fit_transform(np.array(dataframeNew).reshape(-1,1))

dataframeNew.shape


# Preparing DataSets for Training and Test

train_size=int(len(dataframeNew)*0.80)
test_size=len(dataframeNew)-train_size
training_data,testing_data=dataframeNew[0:train_size,:],dataframeNew[train_size:len(dataframeNew),:1]

def create_dataset_for_training_test(dataset_for_creation,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset_for_creation)-time_step-1):
        a = dataset_for_creation[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset_for_creation[i+time_step,0])
    return np.array(dataX),np.array(dataY)


timesteps = 100
X_train,y_train=create_dataset_for_training_test(training_data,timesteps)
X_test,y_test=create_dataset_for_training_test(testing_data,timesteps)

print(X_train)
#plt.show()

model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,dropout=0.3, recurrent_dropout=0.3,return_sequences=True))
model.add(LSTM(50,dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='rmsprop')
model.summary()

model.fit(X_train,y_train,validation_data=(X_test, y_test),epochs=370,batch_size=64,verbose=1)
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

import math
from sklearn.metrics import mean_squared_error

math.sqrt(mean_squared_error(y_train, train_predict))

math.sqrt(mean_squared_error(y_test, test_predict))

look_back_steps_in_time=100

trainPredictPlot=np.empty_like(dataframeNew)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back_steps_in_time:len(train_predict)+look_back_steps_in_time,:]=train_predict
testPredictPlot=np.empty_like(dataframeNew)
testPredictPlot[:,:]=np.nan
testPredictPlot[len(train_predict)+(look_back_steps_in_time*2)+1:len(dataframeNew)-1,:]=test_predict
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

















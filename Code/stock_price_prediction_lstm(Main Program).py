import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from IPython.display import display
import sklearn
import tensorflow
#import Theano


data =pd.read_csv("google.csv")

stocks=data
item = []
open = []
close = []
volume = []

    
i_counter = 0
for i in range(len(data) - 1, -1, -1):
    item.append(i_counter)
    open.append(data['Open'][i])
    volume.append(data['Volume'][i])
    close.append(data['Close'][i])
    i_counter += 1

   
stocks = pd.DataFrame()

   
stocks['Item'] = item
stocks['Open'] = open
stocks['Volume'] = pd.to_numeric(volume)
stocks['Close'] = pd.to_numeric(close)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical = ['Open', 'Close', 'Volume']
stocks[numerical] = scaler.fit_transform(stocks[numerical])
print(stocks)

fig, ax = plt.subplots()
ax.plot(stocks['Item'], stocks['Close'], '#0A7388')


ax.set_title('Google Trading')


plt.ylabel('Price USD')
plt.xlabel('Trading Days')

plt.show()

stocks.to_csv('google_preprocessed.csv',index= False)


from keras.metrics import mean_squared_error
import lstm, time 





stocks = pd.read_csv('google_preprocessed.csv')
#print(stocks)
stocks_data = stocks.drop(['Item'], axis =1)
print(stocks_data)

#display(stocks_data.head())

X_train, X_test,y_train, y_test = lstm.train_test_split_lstm(stocks_data)

unroll_length = 50
X_train = lstm.unroll(X_train, unroll_length)
X_test = lstm.unroll(X_test, unroll_length)
y_train = y_train[-X_train.shape[0]:]
y_test = y_test[-X_test.shape[0]:]

print("x_train", X_train.shape)
print("y_train", y_train.shape)
print("x_test", X_test.shape)
print("y_test", y_test.shape)
#print(X_train)
#print(y_train)
#print(X_train.shape[-1])



model = lstm.build_basic_model(input_dim = X_train.shape[-1],output_dim = unroll_length, return_sequences=True)


start = time.time()
model.compile(loss='mean_squared_error', optimizer='adam')
print('compilation time : ', time.time() - start)

model.fit(X_train,y_train,epochs=10)

predictions = model.predict(X_test)

fig = plt.figure()
ax = fig.add_subplot(111)

 
plt.ylabel('Price USD')
plt.xlabel('Trading Days')

   

plt.plot(y_test, '#00FF00', label='Actual Close')
plt.plot(predictions, '#0000FF', label='Predicted Close')


ax.set_title('Google Trading vs Prediction')
ax.legend(loc='upper left')


plt.show()

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))
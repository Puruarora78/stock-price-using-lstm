import os

# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1785926084.877381   15732 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


import yfinance as yf

ticker = input("Enter ticker of the stock")
ticker = ticker.upper()
start = str(input(f"Enter the date in format (YYYY-MM-DD) from which date sata will be used to train your model (recommended 3 to 6 years) : "))
end = str(input(f"Enter end date same format : "))
try:
    start = datetime.strptime(start, "%Y-%m-%d")
    end = datetime.strptime(end, "%Y-%m-%d")
except:
    print("invalid date format")
    print("please enter the date in the format of YYYY-MM-DD")
    exit()

data = yf.download(ticker , start = start , end = end)
if data.empty:
    print("Invalid ticker or no data available.")
    exit()

if len(data) < 200:
    print("Not enough historical data to train the model.")
    print(f"data available for {len(data)} days please try again")
    exit()

# to remove the unneccesary index
data.columns = data.columns.droplevel(1)

print(data.head())
print(data.info())
print(data.describe())

plt.figure(figsize =(12,6))
plt.plot(data.index, data['Open'], label = "Open", color = "orange")
plt.plot(data.index, data['Close'], label = "Close", color = "green")
plt.title("opening and closing price over time")
plt.legend()
plt.savefig("images/actual_data.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize =(12,6))
plt.plot(data.index, data['Volume'], label = "Date-Volume", color = "red")
plt.title("volume over time")
plt.savefig("images/volume_over_time.png", dpi=300, bbox_inches="tight")
plt.show()

# correlation between features
num_data = data.select_dtypes(include= 'number')
plt.figure(figsize=(10,8))
sns.heatmap(num_data.corr() ,annot= True,fmt = ".2f", cmap = "coolwarm")
plt.title("correlation between features")
plt.savefig("images/heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
print(num_data.corr())

stock_close = data.filter(['Close'])
dataset = stock_close.values
data_L = int(np.ceil(len(data)*0.80))

scale = StandardScaler()
scaled_data = scale.fit_transform(dataset)

train_data = scaled_data[:data_L]

x = []
y = []

for i in range(30,len(train_data)):
    x.append(train_data[i-30:i,0])
    y.append(train_data[i,0])

x = np.array(x)
y = np.array(y)

x = np.reshape(x,(x.shape[0],x.shape[1],1)) 

model = keras.models.Sequential()


# layers
model.add(keras.layers.LSTM(64, return_sequences= True , input_shape = (x.shape[1],1)))
model.add(keras.layers.LSTM(64, return_sequences= False))
model.add(keras.layers.Dense(128, activation= "relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer= "adam",loss= "mae",metrics= [keras.metrics.RootMeanSquaredError()])

training = model.fit(x,y,epochs=20,batch_size=32)

test_data = scaled_data[data_L - 30 :]
x_test = []
y_test = dataset[data_L : ]


for i in range(30,len(test_data)):
    x_test.append(test_data[i-30:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

future_price = model.predict(x_test)
future_price = scale.inverse_transform(future_price)

train = data[:data_L]
test = data[data_L:]

copy = test.copy()

# value used in plt.plot 3rd one
test["FuturePrice"] = future_price


# graph
plt.figure(figsize=(12,8))
plt.plot(train.index,train['Close'],label= "trained_data", color= "red")
plt.plot(test.index,test['Close'],label= "tested data", color = "green")
plt.plot(test.index,test["FuturePrice"],label= "predicted price", color = "blue")
plt.title("price pridiction usind lstm")
plt.xlabel("date")
plt.ylabel("price")
plt.legend()
plt.savefig("images/pridicted_data.png", dpi=300, bbox_inches="tight")
plt.show()

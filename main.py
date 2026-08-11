import os

# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1785926084.877381   15732 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


import yfinance as yf

ticker = input("Enter ticker of the stock : ")
ticker = ticker.upper()
start = str(input(f"Enter the date in format (YYYY-MM-DD) from which date sata will be used to train your model (recommended 3 to 6 years) : "))
end = str(input(f"Enter end date smae format : "))
if start >= end :
    print("please enter the date correctly")
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

if len(data) < 500:
    print("Not enough historical data to train the model.")
    print(f"data available for {len(data)} days please try again")
    exit()

# too many index :/ :/ :/ :/ :/ :/ :/ :/
if isinstance(data.columns,pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

data['sma_20'] = data['Close'].rolling(20).mean()
# print(data[['Close','sma_20']].head(25))
data =data.dropna()

plt.figure(figsize= (12,9))
plt.plot(data['Close'],label = "close" ,color = "red")
plt.plot(data['sma_20'],label = "sma" ,color = "purple")
plt.legend()
plt.show()


# print(data.head())
# print(data.info())
# print(data.describe())


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

features = data[['Close' , 'sma_20']]
target = data[['Close']]
data_L = int(np.ceil(len(data)*0.80))

train_features = features.iloc[:data_L]
test_features = features.iloc[data_L:]

train_target = target.iloc[:data_L]
test_target = target.iloc[data_L:]

TF_scaler = MinMaxScaler()
TT_scaler = MinMaxScaler() 
train_features_scaled = TF_scaler.fit_transform(train_features)
test_features_scaled = TF_scaler.transform(test_features)
train_target_scaled = TT_scaler.fit_transform(train_target)
test_target_scaled = TT_scaler.transform(test_target)

x_train = []
y_train = []

for i in range(30,len(train_features_scaled)):
    x_train.append(train_features_scaled[i-30:i,:])
    y_train.append(train_target_scaled[i,0])

x_train = np.array(x_train)
y_train = np.array(y_train)


model = keras.models.Sequential()


# layers
model.add(keras.layers.LSTM(64, return_sequences= True , input_shape = (x_train.shape[1],x_train.shape[2])))
model.add(keras.layers.LSTM(64, return_sequences= False))
model.add(keras.layers.Dense(128, activation= "relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer= "adam",loss= "mae",metrics= [keras.metrics.RootMeanSquaredError()])

training = model.fit(x_train,y_train,epochs=20,batch_size=32)

test_input = np.concatenate((train_features_scaled[-30:],test_features_scaled))
x_test = []
for i in range(30,len(test_input)):
    x_test.append(test_input[i-30:i,:])

# 30 previous days of close and sma20 data
x_test = np.array(x_test)
# current prediction close data
y_test = test_target_scaled

# x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

future_price = model.predict(x_test)
future_price = TT_scaler.inverse_transform(future_price)
actual_price = TT_scaler.inverse_transform(y_test.reshape(-1,1))

# for aesthetic
print("\n========== Model Evaluation ==========")
mae = mean_absolute_error(future_price,actual_price)
print(f"mae is : {mae:.2f}")

rmse = root_mean_squared_error(future_price,actual_price)
print(f"rmse is : {rmse:.2f}")
print("======================================")

# graph
plt.figure(figsize=(12,8))
plt.plot(train_features.index,train_features["Close"],label= "trained price", color= "red")
plt.plot(test_features.index,test_features["Close"],label= "desired actual price", color = "green")
plt.plot(test_features.index,test_target["Close"],label= "predicted price", color = "blue")
plt.title("price pridiction usind lstm")
plt.xlabel("date")
plt.ylabel("price")
plt.legend()
plt.savefig("images/pridicted_data.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12,8))
plt.plot(test_features.index,test_features["Close"],label= "desired actual price", color = "green")
plt.plot(test_features.index,future_price,label= "predicted price", color = "blue")
plt.title("comparison")
plt.legend()
plt.show()
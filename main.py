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

features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'sma_20']]
target = data[['Close']]
train_data = int(np.ceil(len(data)*0.70))
val_data = int(np.ceil(len(data)*0.80))

train_features = features.iloc[:train_data]
val_features = features.iloc[train_data:val_data]
test_features = features.iloc[val_data:]

train_target = target.iloc[:train_data]
val_target = target.iloc[train_data:val_data]
test_target = target.iloc[val_data:]

TF_scaler = MinMaxScaler()
TT_scaler = MinMaxScaler() 
train_features_scaled = TF_scaler.fit_transform(train_features)
val_features_scaled = TF_scaler.transform(val_features)
test_features_scaled = TF_scaler.transform(test_features)

train_target_scaled = TT_scaler.fit_transform(train_target)
val_target_scaled = TT_scaler.transform(val_target)
test_target_scaled = TT_scaler.transform(test_target)

# sliding window for training data x,y
x_train = []
y_train = []
for i in range(30,len(train_features_scaled)):
    x_train.append(train_features_scaled[i-30:i,:])
    y_train.append(train_target_scaled[i,0])
x_train = np.array(x_train)
y_train = np.array(y_train)


# sliding window for val
val_data_input = np.concatenate((train_features_scaled[-30:],val_features_scaled))
val_data_x = []
for i in range(30,len(val_data_input)):
    val_data_x.append(val_data_input[i-30:i,:])
val_data_x = np.array(val_data_x)
val_data_y = val_target_scaled


model = keras.models.Sequential()


# layers
model.add(keras.layers.LSTM(64, return_sequences= True , input_shape = (x_train.shape[1],x_train.shape[2])))
model.add(keras.layers.LSTM(64, return_sequences= False))
model.add(keras.layers.Dense(128, activation= "relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer= "adam",loss= "mae",metrics= [keras.metrics.RootMeanSquaredError()])

early_stop = keras.callbacks.EarlyStopping(monitor= "val_loss" ,patience= 3 ,restore_best_weights=True)

training = model.fit(x_train,y_train,epochs=20,batch_size=32,validation_data=(val_data_x,val_data_y),callbacks=[early_stop])

plt.plot(training.history["loss"], label="Training Loss")
plt.plot(training.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()

# sliding window for current data it will predict whats the prediction from current data values of sma20,close
test_input = np.concatenate((val_features_scaled[-30:],test_features_scaled))
x_test = []
for i in range(30,len(test_input)):
    x_test.append(test_input[i-30:i,:])
x_test = np.array(x_test)
y_test = test_target_scaled


# x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

future_price = model.predict(x_test)
future_price = TT_scaler.inverse_transform(future_price)
actual_price = TT_scaler.inverse_transform(y_test.reshape(-1,1))

#   ----------------- naive baseline --------------------- 
data_for_bl = np.concatenate((val_target["Close"].values[-1:],test_target["Close"].values[:-1]))
test_values = test_target["Close"].values
print(mean_absolute_error(data_for_bl,test_values))
print(root_mean_squared_error(data_for_bl,test_values))


# for aesthetic
print("\n========== Model Evaluation ==========")
mae = mean_absolute_error(future_price,actual_price)
print(f"mae for LSTM is : {mae:.2f}")

rmse = root_mean_squared_error(future_price,actual_price)
print(f"rmse for LSTM is : {rmse:.2f}")

mae = mean_absolute_error(data_for_bl,test_values)
print(f"mae for Naive Baseline is : {mae:.2f}")

rmse = root_mean_squared_error(data_for_bl,test_values)
print(f"rmse for Naive Baseline is : {rmse:.2f}")
print("======================================")

# graph
plt.figure(figsize=(12,8))
plt.plot(train_features.index,train_features["Close"],label= "trained price", color= "red")
plt.plot(test_features.index,test_features["Close"],label= "desired actual price", color = "green")
plt.plot(val_features.index,val_features["Close"],label= "given val price", color = "purple")
plt.plot(test_features.index,future_price,label= "predicted price", color = "blue")
plt.title("price pridiction usind lstm")
plt.xlabel("date")
plt.ylabel("price")
plt.legend()
plt.savefig("images/pridicted_data.png", dpi=300, bbox_inches="tight")
# plt.show()

plt.figure(figsize=(12,8))
plt.plot(test_features.index,test_features["Close"],label= "desired actual price", color = "green")
plt.plot(test_features.index,future_price,label= "predicted price", color = "blue")
plt.plot(test_features.index,data_for_bl,label= "naive baseline", color = "red")
plt.title("comparison")
plt.legend()
plt.show()
import os

# WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
# I0000 00:00:1785926084.877381   15732 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.dates as mdates

import yfinance as yf

# download data
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
# data downloaded



# too many index :/ :/ :/ :/ :/ :/ :/ :/
if isinstance(data.columns,pd.MultiIndex):
    data.columns = data.columns.droplevel(1)



# data --
# data['sma_20'] = data['Close'].rolling(20).mean()
# print(data[['Close','sma_20']].head(25))
data['Open-Close-ratio-lag-1'] = np.log(data['Open']/data['Close'].shift(1))
data['High-Low-ratio'] = np.log(data['High']/data['Low'])
data['Volume-lag-1'] = np.log(data['Volume'].shift(1))
data['price_change'] = data['Close'].pct_change()
data['log_return'] = np.log(data['Close']/data['Close'].shift(1))
data['volatility_change'] = data['price_change'].rolling(20).std()
data['price_change_lag_1'] = data['price_change'].shift(1)
data['Volume_lag_1'] = data['Volume'].shift(1)
data['volatility_lag_1'] = data['volatility_change'].shift(1)



# loss and gain during training and validation
diff = data['Close'].diff()
loss = -diff.clip(upper=0)
gain = diff.clip(lower=0)
avg_loss = loss.rolling(20).mean()
avg_gain = gain.rolling(20).mean()
# relative_strength = avg_gain / avg_loss
# # relative strenth index
# data['rsi'] = 100-(100/(1+relative_strength)) 
for lag in range(1,11):
    data[f'return_lag_{lag}'] = data['log_return'].shift(lag)

data =data.dropna()

# ---------------------LSTM--------------------------
features = data[['Open-Close-ratio-lag-1','High-Low-ratio','Volume-lag-1','volatility_lag_1','price_change_lag_1']]
target = data[['log_return']]
train_data = int(np.ceil(len(data)*0.70))
val_data = int(np.ceil(len(data)*0.80))

test_dates = data.index[val_data:]

train_features = features.iloc[:train_data]
val_features = features.iloc[train_data:val_data]
test_features = features.iloc[val_data:]

train_target = target.iloc[:train_data]
val_target = target.iloc[train_data:val_data]
test_target = target.iloc[val_data:]

TF_scaler = StandardScaler()
TT_scaler = StandardScaler() 
train_features_scaled = TF_scaler.fit_transform(train_features)
val_features_scaled = TF_scaler.transform(val_features)
test_features_scaled = TF_scaler.transform(test_features)

train_target_scaled = TT_scaler.fit_transform(train_target)
val_target_scaled = TT_scaler.transform(val_target)
test_target_scaled = TT_scaler.transform(test_target)



# sliding window for training data x,y
x_train = []
y_train = []
for i in range(60,len(train_features_scaled)):
    x_train.append(train_features_scaled[i-60:i,:])
    y_train.append(train_target_scaled[i,0])
x_train = np.array(x_train)
y_train = np.array(y_train)



# sliding window for val
val_data_input = np.concatenate((train_features_scaled[-60:],val_features_scaled))
val_data_x = []
for i in range(60,len(val_data_input)):
    val_data_x.append(val_data_input[i-60:i,:])
val_data_x = np.array(val_data_x)
val_data_y = val_target_scaled



model = keras.models.Sequential()

# layers
model.add(keras.layers.LSTM(64, return_sequences= True, input_shape = (x_train.shape[1],x_train.shape[2])))
model.add(keras.layers.LSTM(64, return_sequences= False))
model.add(keras.layers.Dense(128, activation= "relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1))

model.summary()
model.compile(optimizer= "adam",loss= "mae",metrics= [keras.metrics.RootMeanSquaredError()])

early_stop = keras.callbacks.EarlyStopping(monitor= "val_loss" ,patience= 3 ,restore_best_weights=True)

training = model.fit(x_train,y_train,epochs=20,batch_size=32,validation_data=(val_data_x,val_data_y),callbacks=[early_stop])



# sliding window for current data it will predict whats the prediction from current data values
test_input = np.concatenate((val_features_scaled[-60:],test_features_scaled))
x_test = []
for i in range(60,len(test_input)):
    x_test.append(test_input[i-60:i,:])
x_test = np.array(x_test)
y_test = test_target_scaled


future_return = model.predict(x_test)
future_return = TT_scaler.inverse_transform(future_return).flatten()
actual_price = data['Close'].iloc[val_data:].values

previous_1_price = data['Close'].iloc[val_data-1:val_data+len(test_features_scaled)-1].values
predicted_price = previous_1_price * np.exp(future_return)






#   ----------------- naive baseline --------------------- 
data_for_bl = np.concatenate((data["Close"].iloc[val_data-1:val_data].values[-1:],data["Close"].iloc[val_data:-1].values))
test_values = data["Close"].iloc[val_data:].values






#    ------------------ sarima data -----------------------
train_sarima = data['log_return'].iloc[:train_data]
val_sarima = data['log_return'].iloc[train_data:val_data]
test_sarima = data['log_return'].iloc[val_data:]

final_train_data = np.concatenate([train_sarima,val_sarima])

sarima_model = SARIMAX(
    final_train_data,
    order= (1,0,0),
    seasonal_order= (1,0,1,5),
    enforce_invertibility=False,
    enforce_stationarity=False
)

sarima_result = sarima_model.fit(disp=False)
# print(sarima_result.summary())
sarima_current = sarima_result


# residual check 
# lb_1 = acorr_ljungbox(
#     residual_1,
#     lags = [5,10,15,20],
#     return_df= True
# )
# lb_2 = acorr_ljungbox(
#     resid_2,
#     lags= [5,10,15,20],
#     return_df= True
# )
# print(f'lb_1 is : {lb_1}')
# print(f'lb_2 is : {lb_2}')




#no longer needed val 
# sarima_val_prediction = []
# sarima_current = sarima_result
# for i in val_sarima:
#     forecast = sarima_current.forecast(steps = 1)
#     sarima_val_prediction.append(forecast.iloc[0])
#     sarima_current = sarima_current.append([i],refit = False)
# sarima_val_prediction = np.array(sarima_val_prediction)

# sarima_val_actual_lag_1 = data['Close'].iloc[train_data-1:val_data-1]
# sarima_val_prediction_price = sarima_val_actual_lag_1*np.exp(sarima_val_prediction)

# val_data_close = data['Close'].iloc[train_data:val_data]
# mae = mean_absolute_error(sarima_val_prediction_price,val_data_close)
# print(f'mae for sarima validation is : {mae: .6f}')

# rmse = root_mean_squared_error(sarima_val_prediction_price,val_data_close)
# print(f'rsme for sarima validation is : {rmse: .6f}')


sarima_test_prediction = []
for i in test_sarima:
    forecast = sarima_current.forecast(steps= 1)
    sarima_test_prediction.append(forecast[0])
    sarima_current = sarima_current.append([i],refit= False)
sarima_test_prediction = np.array(sarima_test_prediction)

sarima_predicted_price = previous_1_price*np.exp(sarima_test_prediction)

# stationar test 
# adf_result = adfuller(data['log_return'])

# print("\n========== ADF Stationarity Test ==========")
# print(f'statistics are      : {adf_result[0]: .6f}')
# print(f'p value is          : {adf_result[1]: .6f}')
# print(f'critical values are :')
# for key,value in adf_result[4].items():
#     print(f'{key} : {value: .2f}')



#    ------------------- xgboost --------------------------
xgboost_features = data[
    [
        'Open', 
        # 'High', 
        # 'Low',
        'price_change_lag_1',
        'Volume_lag_1',
        'volatility_lag_1',
        "return_lag_1",
        "return_lag_2",
        "return_lag_3",
        "return_lag_4",
        "return_lag_5",
        "return_lag_6",
        "return_lag_7",
        "return_lag_8",
        "return_lag_9",
        "return_lag_10"
    ]
]

xgboost_target = data['log_return']

xgb_x_train = xgboost_features.iloc[:train_data]
xgb_x_val = xgboost_features.iloc[train_data:val_data]
xgb_x_test = xgboost_features.iloc[val_data:]


xgb_y_train = xgboost_target.iloc[:train_data]
xgb_y_val = xgboost_target.iloc[train_data:val_data]
xgb_y_test = xgboost_target.iloc[val_data:]

xgb_model = XGBRegressor(
    n_estimators = 200,
    learning_rate = 0.05,
    max_depth = 3,
    random_state = None
)

xgb_model.fit(
    xgb_x_train,
    xgb_y_train,
    eval_set = [(xgb_x_val,xgb_y_val)],
    verbose = False
)

xgb_prediction = xgb_model.predict(xgb_x_test)
xgb_predicted_price = previous_1_price*np.exp(xgb_prediction)


# for aesthetic
print("\n========== Model Evaluation ==========")
mae = mean_absolute_error(predicted_price,actual_price)
print(f"mae for LSTM is : {mae:.6f}")

rmse = root_mean_squared_error(predicted_price,actual_price)
print(f"rmse for LSTM is : {rmse:.6f}")

mae = mean_absolute_error(data_for_bl,test_values)
print(f"mae for Naive Baseline is : {mae:.6f}")

rmse = root_mean_squared_error(data_for_bl,test_values)
print(f"rmse for Naive Baseline is : {rmse:.6f}")

mae = mean_absolute_error(xgb_predicted_price,actual_price)
print(f'mae for XGBoost is : {mae: .6f}')

rmse = root_mean_squared_error(xgb_predicted_price,actual_price)
print(f'rmse for XGBoost is : {rmse: .6f}')

mae = mean_absolute_error(sarima_predicted_price,actual_price)
print(f'mae for sarima is : {mae: .6f}')

rmse = root_mean_squared_error(sarima_predicted_price,actual_price)
print(f'rsme for sarima is : {rmse: .6f}')

print("======================================")

# # graph
# plt.figure(figsize= (12,9))
# plt.plot(data['Close'],label = "close" ,color = "red")
# plt.plot(data['sma_20'],label = "sma" ,color = "purple")
# plt.legend()
# plt.show()

# plt.figure(figsize =(12,6))
# plt.plot(data.index, data['Open'], label = "Open", color = "orange")
# plt.plot(data.index, data['Close'], label = "Close", color = "green")
# plt.title("opening and closing price over time")
# plt.legend()
# plt.savefig("images/actual_data.png", dpi=600, bbox_inches="tight")
# plt.show()

# plt.figure(figsize =(12,6))
# plt.plot(data.index, data['Volume'], label = "Date-Volume", color = "red")
# plt.title("volume over time")
# plt.savefig("images/volume_over_time.png", dpi=600, bbox_inches="tight")
# plt.show()

# correlation between features
# num_data = data.select_dtypes(include= 'number')
# plt.figure(figsize=(10,8))
# sns.heatmap(num_data.corr() ,annot= True,fmt = ".2f", cmap = "coolwarm")
# plt.title("correlation between features")
# plt.savefig("images/heatmap.png", dpi=600, bbox_inches="tight")
# plt.show()
# print(num_data.corr())

plt.plot(training.history["loss"], label="Training Loss")
plt.plot(training.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("images/loss_val-loss_data.png", dpi=600, bbox_inches="tight")
plt.show()

# plt.figure(figsize=(12,8))
# plt.plot(train_features.index,train_features["Close"],label= "trained price", color= "red")
# plt.plot(test_features.index,test_features["Close"],label= "desired actual price", color = "green")
# plt.plot(val_features.index,val_features["Close"],label= "given val price", color = "purple")
# plt.plot(test_features.index,predicted_price,label= "predicted price", color = "blue")
# plt.title("price pridiction usind lstm")
# plt.xlabel("date")
# plt.ylabel("price")
# plt.legend()
# plt.savefig("images/pridicted_data.png", dpi=600, bbox_inches="tight")
# # plt.show()

# plt.figure(figsize=(12,8))
# plt.plot(test_features.index,test_features["Close"],label= "desired actual price", color = "green")
# plt.plot(test_features.index,predicted_price,label= "predicted price", color = "blue")
# plt.plot(test_features.index,data_for_bl,label= "naive baseline", color = "red")
# plt.title("comparison")
# plt.legend()
# plt.show()


plt.figure(figsize=(12,8))
plt.plot(test_dates,actual_price,label = "Actual Price")
plt.plot(test_dates,predicted_price,label = "LSTM predicted Price")
plt.plot(test_dates,data_for_bl,label = "Naive Baseline Predicted Price")
plt.plot(test_dates,xgb_predicted_price,label = "XGBoost Predicted Price")
plt.plot(test_dates,sarima_predicted_price,label = "Sarima Predicted Price")
plt.xlabel('Dates')
plt.ylabel("Price")
plt.title("Actual-Predicted Price")

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=45)
                
plt.legend()
plt.tight_layout()
plt.savefig(f"images/actual-predicted-price/{ticker}_Actual-Predicted Price_For_{len(test_dates)}_Days.png", dpi = 600 , bbox_inches = "tight")
plt.show()

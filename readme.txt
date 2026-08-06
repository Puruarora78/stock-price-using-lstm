Hi
My name is Puru Arora.
this is a stock prediction model using lstm(long-short term memory) neural network using keras.
i have use tesla stocks as an example.
this model pridicts Tesla's stock closing price. 
i have downloaded the data from yfinance.

features :-
1 download historical data from yfinance using ticker
2 data visualization using matplotlib
3 correlation using seaborn
4 Pridicted price using LSTM
5 visualization of predicted and actual stock close price

tech used :-
1 python
2 keras
3 matplotlib
4 pandas
5 numpy
6 seaborn
7 scikit-learn 
8 yahoo finance
9 datetime

installation :-
Clone the repository
https://github.com/Puruarora78/stock-price-using-lstm.git


Move into the project
cd stock-price-using-lstm

Install dependencies
pip install -r requirements.txt

Running the Project
python main.py

<------------------------------------------------------------------------------------------->

results when tested

corre :-
Price      Close      High       Low      Open    Volume
Price                                                   
Close   1.000000  0.999520  0.999556  0.998908 -0.230520
High    0.999520  1.000000  0.999455  0.999579 -0.226208
Low     0.999556  0.999455  1.000000  0.999477 -0.236867
Open    0.998908  0.999579  0.999477  1.000000 -0.231786
Volume -0.230520 -0.226208 -0.236867 -0.231786  1.000000

![Heatmap](images/heatmap.png)

<------------------------------------------------------------------------------------------->

Actual stock price visualization :-

![Stock Price](images/stock_price.png)

<------------------------------------------------------------------------------------------->


model summary :-

<class 'pandas.DataFrame'>
DatetimeIndex: 2520 entries, 2016-04-25 to 2026-05-01
Data columns (total 5 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Close   2520 non-null   float64
 1   High    2520 non-null   float64
 2   Low     2520 non-null   float64
 3   Open    2520 non-null   float64
 4   Volume  2520 non-null   int64  
dtypes: float64(4), int64(1)
memory usage: 118.1 KB
None
Price        Close         High          Low         Open        Volume
count  2520.000000  2520.000000  2520.000000  2520.000000  2.520000e+03
mean    162.174415   165.714049   158.498049   162.218113  1.148505e+08
std     136.085522   139.006360   133.131470   136.205619  7.268475e+07
min      11.931333    12.315333    11.799333    12.073333  2.489250e+07
25%      21.262833    21.608500    20.847333    21.194500  7.033088e+07
50%     176.414993   179.600006   172.930000   175.680000  9.570460e+07
75%     259.166672   264.037506   252.827503   258.147491  1.315992e+08
max     489.880005   498.829987   485.329987   489.880005  9.140820e+08
Price      Close      High       Low      Open    Volume
Price                                                   
Close   1.000000  0.999520  0.999556  0.998908 -0.230520
High    0.999520  1.000000  0.999455  0.999579 -0.226208
Low     0.999556  0.999455  1.000000  0.999477 -0.236867
Open    0.998908  0.999579  0.999477  1.000000 -0.231786
Volume -0.230520 -0.226208 -0.236867 -0.231786  1.000000


Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                          │ (None, 30, 64)              │          16,896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_1 (LSTM)                        │ (None, 64)                  │          33,024 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │           8,320 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 1)                   │             129 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 58,369 (228.00 KB)
 Trainable params: 58,369 (228.00 KB)
 Non-trainable params: 0 (0.00 B)

<------------------------------------------------------------------------------------------->

model prediction :-

![Prediction](images/prediction.png)


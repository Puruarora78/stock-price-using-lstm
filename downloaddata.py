import yfinance as yf
import pandas as pd

ticker = 'TSLA'

data = yf.download(ticker , start = '2016-04-23' , end = '2026-05-03')

print(data.head())

print("done")
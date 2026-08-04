from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib as plt
import seaborn as sns
import os
from datetime import datetime

data = pd.read_csv('TESLA stock.csv')

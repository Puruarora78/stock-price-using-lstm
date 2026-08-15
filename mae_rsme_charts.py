import numpy as np
import matplotlib.pyplot as plt

datasets = [
    "AAPL 3y",
    "AAPL 10y",
    "TSLA 3y",
    "TSLA 10y",
    "MSFT 3y",
    "MSFT 10y"
]

naive_mae = [
    2.261279,
    2.479543,
    9.177741,
    8.267410,
    4.195265,
    4.079825
]

lstm_mae = [
    2.309172,
    2.487076,
    9.229435,
    8.381798,
    4.202180,
    4.072076
]

xgb_mae = [
    2.379145,
    2.662618,
    10.635090,
    8.646171,
    4.360533,
    4.834670
]

sarima_mae = [
    2.201945,
    2.529249,
    9.414005,
    8.272998,
    4.241584,
    4.076552
]


# -----------------------------------------rmse-----------------------------------------
naive_rmse = [
    3.180520,
    3.663767,
    12.003227,
    11.440186,
    5.894364,
    5.443339
]

lstm_rmse = [
    3.203036,
    3.673992,
    12.090004,
    11.689721,
    5.897566,
    5.466045
]

xgb_rmse = [
    3.371673,
    3.857736,
    13.705428,
    11.975411,
    6.084945,
    6.178767
]

sarima_rmse = [
    3.147529,
    3.715767,
    12.294855,
    11.446678,
    5.961695,
    5.436848
]


# ----------------------------------graphs-----------------------------------


#  mae graph

x = np.arange(len(datasets))
width = 0.2

plt.figure(figsize=(12, 6))

bars1 = plt.bar(x - 1.5*width, naive_mae, width, label="Naive")
bars2 = plt.bar(x - 0.5*width, lstm_mae, width, label="LSTM")
bars3 = plt.bar(x + 0.5*width, xgb_mae, width, label="XGBoost")
bars4 = plt.bar(x + 1.5*width, sarima_mae, width, label="SARIMA")


plt.bar_label(bars1, fmt="%.2f", padding=3)
plt.bar_label(bars2, fmt="%.2f", padding=3)
plt.bar_label(bars3, fmt="%.2f", padding=3)
plt.bar_label(bars4, fmt="%.2f", padding=3)

plt.xticks(x, datasets)
plt.xlabel("Dataset")
plt.ylabel("MAE")
plt.title("MAE Comparison Across Stocks and Historical Windows")

plt.legend()
plt.tight_layout()
plt.savefig("images/mae_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


#      rmse graph

x = np.arange(len(datasets))
width = 0.2

plt.figure(figsize=(12, 6))

bars5 = plt.bar(x - 1.5*width, naive_rmse, width, label="Naive")
bars6 = plt.bar(x - 0.5*width, lstm_rmse, width, label="LSTM")
bars7 = plt.bar(x + 0.5*width, xgb_rmse, width, label="XGBoost")
bars8 = plt.bar(x + 1.5*width, sarima_rmse, width, label="SARIMA")

plt.bar_label(bars5, fmt="%.2f", padding=3)
plt.bar_label(bars6, fmt="%.2f", padding=3)
plt.bar_label(bars7, fmt="%.2f", padding=3)
plt.bar_label(bars8, fmt="%.2f", padding=3)

plt.xticks(x, datasets)
plt.xlabel("Dataset")
plt.ylabel("RMSE")
plt.title("RMSE Comparison Across Stocks and Historical Windows")

plt.legend()
plt.tight_layout()
plt.savefig("images/rmse_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
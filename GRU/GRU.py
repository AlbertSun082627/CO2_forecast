import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential 
from keras.layers import Dense, GRU, Dropout
from keras.regularizers import l1, l2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 讀取訓練數據集
dataset_train = pd.read_csv("C:/Users/Sun/Desktop/二氧化碳濃度預測/模型/CO2_History.csv", encoding='utf-8')
training_set = dataset_train.iloc[:,2:3].values

# 特徵縮放
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# 創建訓練數據集
X_train = []
y_train = []
for i in range(60, 742):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# 初始化GRU模型
regressor = Sequential()

# 添加第一個GRU層和一些Dropout正則化
regressor.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# 添加第二個GRU層和一些Dropout正則化
regressor.add(GRU(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# 添加第三個GRU層和一些Dropout正則化
regressor.add(GRU(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# 添加第四個GRU層和一些Dropout正則化
regressor.add(GRU(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# 添加第五個GRU層和一些Dropout正則化
regressor.add(GRU(units=50))
regressor.add(Dropout(0.2))

# 添加輸出層
regressor.add(Dense(units = 1))

# 編譯模型
regressor.compile(optimizer='adam', loss='mean_squared_error')

# 適配模型
regressor.fit(X_train, y_train, epochs=100, batch_size=8)

# 讀取測試數據集
dataset_test = pd.read_csv("C:/Users/Sun/Desktop/二氧化碳濃度預測/模型/CO2_thisyear.csv", encoding='utf-8')
real_stock_price = dataset_test.iloc[:,2:3].values

# 整合數據集
dataset_total = pd.concat([dataset_train["monthly average"], dataset_test["monthly average"]], axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# 創建測試數據集
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 預測
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# 計算預測誤差
absolute_errors = abs(predicted_stock_price - real_stock_price)
average_error = np.mean(absolute_errors)

# 計算均方根誤差（RMSE）
rmse = np.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

# 計算平均絕對誤差（MAE）
mae = mean_absolute_error(real_stock_price, predicted_stock_price)

# 計算R平方
r_squared = r2_score(real_stock_price, predicted_stock_price)

print("Average Error (AE):", average_error)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R^2):", r_squared)

# 繪製結果
plt.plot(real_stock_price, color = "red", label = "Actual monthly average")
plt.plot(predicted_stock_price, color = "blue", label = "Predicted monthly average")
plt.title("carbon dioxide concentration")
plt.xlabel("time")
plt.ylabel("ppm")
plt.legend()
plt.show()

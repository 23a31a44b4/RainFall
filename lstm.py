# =====================================
# 1️⃣ Import Libraries
# =====================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =====================================
# 2️⃣ Load Dataset
# =====================================

df = pd.read_csv("data/cleaned/rainfall_dataset_cleaned.csv")   

FEATURES = [
    "temperature_celsius",
    "humidity",
    "pressure_mb",
    "wind_kph",
    "cloud",
    "uv_index",
    "visibility_km",
    "gust_kph"
]

TARGET = "precip_mm"

# Keep only needed columns
data = df[FEATURES + [TARGET]]

# =====================================
# 3️⃣ Scale Data
# =====================================

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# =====================================
# 4️⃣ Create Sequences (Time Series)
# =====================================

sequence_length = 3   # Use last 3 days

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, :-1])  # All features except target
    y.append(scaled_data[i, -1])                     # Target (precip_mm)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)  # (samples, 3, 8)
print("y shape:", y.shape)

# =====================================
# 5️⃣ Train Test Split
# =====================================

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =====================================
# 6️⃣ Build LSTM Model
# =====================================

model = Sequential()

model.add(LSTM(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# =====================================
# 7️⃣ Train Model
# =====================================

model.fit(X_train, y_train, epochs=18, batch_size=16)


# Save the trained LSTM model
model.save("lstm_model.h5")

# # =====================================
# # 8️⃣ Predictions
# # =====================================

# predictions = model.predict(X_test)

# # Reverse scale only rainfall column
# rainfall_scaler = MinMaxScaler()
# rainfall_scaler.fit(df[[TARGET]])

# predictions = rainfall_scaler.inverse_transform(predictions)
# y_test_actual = rainfall_scaler.inverse_transform(y_test.reshape(-1, 1))

# # =====================================
# # 9️⃣ Evaluation
# # =====================================

# rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
# mae = mean_absolute_error(y_test_actual, predictions)

# print("RMSE:", rmse)
# print("MAE:", mae)

# # =====================================
# # 🔟 Plot Results
# # =====================================

# plt.figure()
# plt.plot(y_test_actual, label="Actual Rainfall")
# plt.plot(predictions, label="Predicted Rainfall")
# plt.legend()
# plt.title("LSTM Rainfall Prediction")
# plt.show()
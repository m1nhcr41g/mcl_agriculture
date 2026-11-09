import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/data_cleaned_full.csv")

def create_advanced_features(data):
    df_fe = data.copy()
    df_fe["ET_Index"] = (df_fe["Temperature_C"] * df_fe["Sunlight_Hours"] * (1 + 0.1 * df_fe["Wind_Speed_kmh"])) / (df_fe["Humidity"] + 1)
    df_fe["Moisture_Deficit"] = (df_fe["Soil_Moisture"] / (df_fe["Rainfall_mm"] + 1))
    df_fe["Heat_Humidity_Ratio"] = (df_fe["Temperature_C"] / (df_fe["Humidity"] + 1))
    df_fe["Thermal_Radiation"] = (df_fe["Temperature_C"] * df_fe["Sunlight_Hours"])
    return df_fe

df_fe = create_advanced_features(df)

# Chia dữ liệu 2D theo thứ tự thời gian: 70% Train - 10% Val - 20% Test
n = len(df_fe)
train_end = int(n * 0.7)
val_end = int(n * 0.8)

train_df = df_fe.iloc[:train_end]
val_df = df_fe.iloc[train_end:val_end]
test_df = df_fe.iloc[val_end:]

# Tách X và y
X_train_raw, y_train_raw = train_df.drop(columns=["Target"]).values, train_df["Target"].values
X_val_raw, y_val_raw = val_df.drop(columns=["Target"]).values, val_df["Target"].values
X_test_raw, y_test_raw = test_df.drop(columns=["Target"]).values, test_df["Target"].values

# Fit Scaler CHỈ trên tập Train để tránh Data Leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

joblib.dump(scaler, "models/scaler_lstm.pkl")

# Sliding Window (Tăng lên 5 timesteps để mở rộng ngữ cảnh thời gian)
TIME_STEPS = 5

def create_sequences(X_data, y_data, time_steps):
    X_seq, y_seq = [], []
    for i in range(len(X_data) - time_steps + 1):
        X_seq.append(X_data[i:i + time_steps])
        y_seq.append(y_data[i + time_steps - 1])
    return np.array(X_seq), np.array(y_seq)

# Tạo mảng 3D độc lập cho 3 tập
X_train, y_train = create_sequences(X_train_scaled, y_train_raw, TIME_STEPS)
X_val, y_val = create_sequences(X_val_scaled, y_val_raw, TIME_STEPS)
X_test, y_test = create_sequences(X_test_scaled, y_test_raw, TIME_STEPS)

# Lưu tập dữ liệu chuỗi
np.savez(
    "data/data_sequence_for_lstm.npz",
    x_train=X_train,
    x_val=X_val,
    x_test=X_test,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test
)
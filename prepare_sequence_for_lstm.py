import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data_cleaned_full.csv")

def create_advanced_features(data):
    df_fe = data.copy()

    df_fe["ET_Index"] = (df_fe["Temperature_C"]* df_fe["Sunlight_Hours"]* (1 + 0.1 * df_fe["Wind_Speed_kmh"])) / (df_fe["Humidity"] + 1)

    df_fe["Moisture_Deficit"] = (df_fe["Soil_Moisture"]/ (df_fe["Rainfall_mm"] + 1))

    df_fe["Heat_Humidity_Ratio"] = (df_fe["Temperature_C"]/ (df_fe["Humidity"] + 1))

    df_fe["Thermal_Radiation"] = (df_fe["Temperature_C"]* df_fe["Sunlight_Hours"])

    return df_fe


df_fe = create_advanced_features(df)

#chia dữ liệu tĩnh ban đầu 2D theo tỷ lệ 80-20
train_size_raw = int(len(df_fe) * 0.8)
train_df = df_fe.iloc[:train_size_raw]
test_df = df_fe.iloc[train_size_raw:]

# Tách X và y cho từng tập riêng biệt
X_train_raw = train_df.drop(columns=["Target"]).values
y_train_raw = train_df["Target"].values

X_test_raw = test_df.drop(columns=["Target"]).values
y_test_raw = test_df["Target"].values

#Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)
joblib.dump(
    scaler,
    "scaler_lstm.pkl"
)

#Sliding Window
TIME_STEPS = 3
def create_sequences(X_data, y_data, time_steps):
    X_seq = []
    y_seq = []
    for i in range(len(X_data) - time_steps+1):
        X_seq.append(X_data[i:i + time_steps])
        y_seq.append(y_data[i + time_steps-1])
    return np.array(X_seq), np.array(y_seq)

# 6. TẠO SLIDING WINDOW 3D ĐỘC LẬP CHO TỪNG TẬP
X_train, y_train = create_sequences(X_train_scaled, y_train_raw, TIME_STEPS)
X_test, y_test = create_sequences(X_test_scaled, y_test_raw, TIME_STEPS)

#Lưu dữ liệu
np.savez(
    "data_sequence_for_lstm.npz",
    x_train=X_train,
    x_test=X_test,

    y_train=y_train,
    y_test=y_test

)

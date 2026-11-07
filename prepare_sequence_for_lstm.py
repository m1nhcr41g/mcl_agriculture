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

#Tách X và y
X = df_fe.drop(columns=["Target"]).values
y = df_fe["Target"].values

#Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(
    scaler,
    "scaler_lstm.pkl"
)

#Sliding Window
TIME_STEPS = 3
def create_sequences(X_data, y_data, time_steps):
    
    X_seq = []
    y_seq = []

    for i in range(len(X_data) - time_steps):

        X_seq.append(X_data[i:i + time_steps])

        y_seq.append(y_data[i + time_steps])

    return np.array(X_seq), np.array(y_seq)


X_seq, y_seq = create_sequences(
    X_scaled,
    y,
    TIME_STEPS
)

#Train/Test Split theo thứ tự

train_size = int(len(X_seq) * 0.8)

X_train = X_seq[:train_size]
X_test = X_seq[train_size:]

y_train = y_seq[:train_size]
y_test = y_seq[train_size:]

#Lưu dữ liệu
np.savez(
    "data_sequence_for_lstm.npz",
    x_train=X_train,
    x_test=X_test,

    y_train=y_train,
    y_test=y_test

)

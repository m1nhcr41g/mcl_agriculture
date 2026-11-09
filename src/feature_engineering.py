import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/data_cleaned_full.csv')
def create_advanced_features(data):
    df_fe = data.copy()
    df_fe['ET_Index'] = (
        df_fe['Temperature_C'] * df_fe['Sunlight_Hours'] * (1 + 0.1 * df_fe['Wind_Speed_kmh'])
    ) / (df_fe['Humidity'] + 1)
    df_fe['Moisture_Deficit'] = df_fe['Soil_Moisture'] / (df_fe['Rainfall_mm'] + 1)
    df_fe['Heat_Humidity_Ratio'] = df_fe['Temperature_C'] / (df_fe['Humidity'] + 1)
    df_fe['Thermal_Radiation'] = df_fe['Temperature_C'] * df_fe['Sunlight_Hours']
    return df_fe
df_fe = create_advanced_features(df)
x = df_fe.drop(columns=['Target'])
y = df_fe['Target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
# train_fe = x_train.copy()
# train_fe['Target'] = y_train
# test_fe = x_test.copy()
# test_fe['Target']=y_test
# train_fe.to_csv('data_train_fe.csv', index=False)
# test_fe.to_csv('data_test_fe.csv', index=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

np.savez(
    'data/data_scaled_fe_for_lstm.npz', 
    x_train=X_train_scaled, 
    x_test=X_test_scaled, 
    y_train=y_train.values, 
    y_test=y_test.values
)
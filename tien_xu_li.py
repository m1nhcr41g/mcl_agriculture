import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('irrigation_prediction.csv')
x = df.drop(columns=['Irrigation_Need'])
y = df['Irrigation_Need']
target_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
y_encoded = y.map(target_mapping)
categorical_cols = x.select_dtypes(include=['object','string']).columns.to_list()
x_encoded = x.copy()
label_encoders  = {}
for col  in categorical_cols:
    le = LabelEncoder()
    x_encoded[col] = le.fit_transform(x_encoded[col])
    label_encoders[col] = le
full_data = x_encoded.copy()
full_data['Target'] = y_encoded
full_data.to_csv('data_cleaned_full.csv', index=False)
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
scaler = StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
train_data = x_train.copy()
train_data['Target'] = y_train
train_data.to_csv('data_train.csv', index=False)
test_data = x_test.copy()
test_data['Target'] = y_test
test_data.to_csv('data_test.csv', index=False)
np.savez('data_scaled_cho_lstm.npz', x_train=x_train_scaled, x_test=x_test_scaled, y_train=y_train.values, y_test=y_test.values)

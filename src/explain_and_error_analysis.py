import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf

data_lstm = np.load("data/data_sequence_for_lstm.npz")
X_test_3d = data_lstm["x_test"]
y_test = data_lstm["y_test"]

scaler = joblib.load("models/scaler_lstm.pkl")

X_test_2d_scaled = X_test_3d[:, -1, :]
X_test_2d_raw = scaler.inverse_transform(X_test_2d_scaled)

df_raw = pd.read_csv("data/data_cleaned_full.csv")

def create_advanced_features(data):
    df_fe = data.copy()
    df_fe["ET_Index"] = (df_fe["Temperature_C"] * df_fe["Sunlight_Hours"] * (1 + 0.1 * df_fe["Wind_Speed_kmh"])) / (df_fe["Humidity"] + 1)
    df_fe["Moisture_Deficit"] = (df_fe["Soil_Moisture"] / (df_fe["Rainfall_mm"] + 1))
    df_fe["Heat_Humidity_Ratio"] = (df_fe["Temperature_C"] / (df_fe["Humidity"] + 1))
    df_fe["Thermal_Radiation"] = (df_fe["Temperature_C"] * df_fe["Sunlight_Hours"])
    return df_fe

df_fe = create_advanced_features(df_raw)
feature_names = [col for col in df_fe.columns if col != "Target"]
target_names = ["Low", "Medium", "High"]

X_test_2d_df = pd.DataFrame(X_test_2d_raw, columns=feature_names)

model_lgbm = joblib.load("models/model_lightgbm.pkl")
model_rf = joblib.load("models/model_random_forest.pkl")
model_lstm = tf.keras.models.load_model("models/model_lstm.keras")

explainer_lgbm = shap.TreeExplainer(model_lgbm)
shap_lgbm = explainer_lgbm(X_test_2d_df)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_lgbm, X_test_2d_df, class_names=target_names, show=False)
plt.title("SHAP Feature Importance - LightGBM", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("reports/shap_summary_lightgbm.png", dpi=300)
plt.close()

explainer_rf = shap.TreeExplainer(model_rf)
shap_rf = explainer_rf(X_test_2d_df)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_rf, X_test_2d_df, class_names=target_names, show=False)
plt.title("SHAP Feature Importance - Random Forest", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("reports/shap_summary_random_forest.png", dpi=300)
plt.close()

baseline_preds = np.argmax(model_lstm.predict(X_test_3d, verbose=0), axis=1)
baseline_acc = np.mean(baseline_preds == y_test)

lstm_importances = []
for f_idx in range(X_test_3d.shape[2]):
    X_permuted = X_test_3d.copy()
    np.random.shuffle(X_permuted[:, :, f_idx])
    perm_preds = np.argmax(model_lstm.predict(X_permuted, verbose=0), axis=1)
    perm_acc = np.mean(perm_preds == y_test)
    lstm_importances.append(baseline_acc - perm_acc)

df_lstm_imp = pd.DataFrame({"Feature": feature_names, "Importance": lstm_importances})
df_lstm_imp = df_lstm_imp.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=df_lstm_imp, hue="Feature", palette="Blues_r", legend=False)
plt.title("Feature Importance - LSTM (Permutation Importance)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig("reports/feature_importance_lstm.png", dpi=300)
plt.close()

pred_lgbm = model_lgbm.predict(X_test_2d_df)
pred_rf = model_rf.predict(X_test_2d_df)
pred_lstm = np.argmax(model_lstm.predict(X_test_3d, verbose=0), axis=1)

df_compare = pd.DataFrame({
    "Actual": y_test,
    "Pred_LightGBM": pred_lgbm,
    "Pred_RandomForest": pred_rf,
    "Pred_LSTM": pred_lstm
})

df_compare["Error_LGBM"] = df_compare["Actual"] != df_compare["Pred_LightGBM"]
df_compare["Error_RF"] = df_compare["Actual"] != df_compare["Pred_RandomForest"]
df_compare["Error_LSTM"] = df_compare["Actual"] != df_compare["Pred_LSTM"]

print("[TỔNG HỢP SỐ MẪU ĐOÁN SAI CHUẨN ĐỒNG BỘ]")
print(f" - LightGBM     : {df_compare['Error_LGBM'].sum()} / {len(y_test)} mẫu")
print(f" - Random Forest: {df_compare['Error_RF'].sum()} / {len(y_test)} mẫu")
print(f" - LSTM         : {df_compare['Error_LSTM'].sum()} / {len(y_test)} mẫu")

hard_samples = df_compare[df_compare["Error_LGBM"] & df_compare["Error_RF"] & df_compare["Error_LSTM"]]
print(f"\nSố lượng mẫu dị biệt (Cả 3 mô hình cùng đoán sai): {len(hard_samples)} mẫu")

df_compare.to_csv("data/error_analysis_comparison_all_models.csv", index=False)
print("Đã lưu bảng phân tích sai số tổng hợp vào: error_analysis_comparison_all_models.csv")

errors_summary = pd.DataFrame({
    "Model": ["LightGBM", "Random Forest", "LSTM"],
    "Error_Count": [df_compare['Error_LGBM'].sum(), df_compare['Error_RF'].sum(), df_compare['Error_LSTM'].sum()]
})

plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Error_Count", data=errors_summary, hue="Model", palette="Reds_d", legend=False)
plt.title("So Sánh Số Lượng Mẫu Dự Đoán Sai Trên Tập Test", fontsize=14, pad=15)
plt.ylabel("Số mẫu đoán sai")
for p in plt.gca().patches:
    plt.gca().annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.5), ha='center')
plt.tight_layout()
plt.savefig("reports/error_comparison_chart.png", dpi=300)
plt.close()
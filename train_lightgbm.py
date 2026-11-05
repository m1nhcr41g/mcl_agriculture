import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)

train_df = pd.read_csv("data_train_fe.csv")
test_df = pd.read_csv("data_test_fe.csv")
X_train = train_df.drop(columns=["Target"])
y_train = train_df["Target"]
X_test = test_df.drop(columns=["Target"])
y_test = test_df["Target"]

print("Kích thước tập huấn luyện:", X_train.shape)
print("Kích thước tập kiểm tra:", X_test.shape)

# 2. KHỞI TẠO MÔ HÌNH LIGHTGBM
model = LGBMClassifier(objective="multiclass",num_class=3,n_estimators=300,learning_rate=0.05,num_leaves=31,random_state=42,n_jobs=-1,verbosity=-1)

# 3. HUẤN LUYỆN MÔ HÌNH
print("\nĐang huấn luyện mô hình LightGBM...")
model.fit(X_train, y_train)
print("Huấn luyện hoàn tất!")

# 4. DỰ ĐOÁN TRÊN TẬP TEST
y_pred = model.predict(X_test)

# 5. ĐÁNH GIÁ MÔ HÌNH
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n========== KẾT QUẢ ĐÁNH GIÁ LIGHTGBM ==========")
print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-score  : {f1:.4f} ({f1 * 100:.2f}%)")

# 6. BÁO CÁO CHI TIẾT TỪNG LỚP
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test,y_pred,labels=[0, 1, 2],target_names=["Low", "Medium", "High"],zero_division=0))

# 7. TRỰC QUAN HÓA KẾT QUẢ ĐÁNH GIÁ
cm = confusion_matrix(y_test,y_pred,labels=[0, 1, 2])
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
scores = [accuracy,precision,recall,f1]
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

colors = ["#4B4E83", "#36788A", "#319B7E", "#80C267"]
bars = axes[0].bar(metrics,scores,color=colors,width=0.8)
axes[0].set_title("Đánh Giá Mô Hình LightGBM Bằng 4 Độ Đo",fontsize=15,fontweight="bold")
axes[0].set_xlabel("Metric", fontsize=12)
axes[0].set_ylabel("Giá Trị (0.0 - 1.0)", fontsize=12)
axes[0].set_ylim(0, 1.15)

for bar, score in zip(bars, scores):axes[0].text(bar.get_x() + bar.get_width() / 2,bar.get_height() + 0.015,f"{score * 100:.2f}%",ha="center",fontsize=11,fontweight="bold")

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=["Low (0)", "Medium (1)", "High (2)"],yticklabels=["Low (0)", "Medium (1)", "High (2)"],ax=axes[1],cbar=True)
axes[1].set_title("Ma Trận Nhầm Lẫn (Confusion Matrix)",fontsize=15,fontweight="bold")
axes[1].set_xlabel("Giá Trị Dự Đoán (Predicted)", fontsize=12)
axes[1].set_ylabel("Giá Trị Thực Tế (Actual)", fontsize=12)

plt.tight_layout()
plt.savefig("evaluation_results_lightgbm.png",dpi=300,bbox_inches="tight")
plt.show()
print("\nĐã lưu biểu đồ đánh giá: evaluation_results_lightgbm.png")
joblib.dump(model, "model_lightgbm.pkl")

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)

# Cố định Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Đọc dữ liệu
data = np.load("data/data_sequence_for_lstm.npz")
X_train, y_train = data["x_train"], data["y_train"]
X_val, y_val = data["x_val"], data["y_val"]
X_test, y_test = data["x_test"], data["y_test"]

print("\nKích thước các tập dữ liệu:")
print("Train      :", X_train.shape)
print("Validation :", X_val.shape)
print("Test       :", X_test.shape)

num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

time_steps = X_train.shape[1]
num_features = X_train.shape[2]

# Kiến trúc mô hình LSTM (Đã cập nhật layer Input chuẩn Keras)
model = Sequential([
    Input(shape=(time_steps, num_features)),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Tính toán Class Weights chuẩn và LÀM MỊN bằng căn bậc 2 (Smoothed Weights)
unique_classes = np.unique(y_train)
raw_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
smoothed_weights = np.sqrt(raw_weights)  # Giảm phạt quá đà cho lớp ít mẫu (High)

class_weight = dict(zip(unique_classes, smoothed_weights))
print("\nClass Weight (Đã làm mịn):", class_weight)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint("models/model_lstm.keras", monitor="val_loss", save_best_only=True, verbose=1)
]

# Huấn luyện mô hình
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=60,
    batch_size=32,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1
)

# Dự đoán và Đánh giá trên tập Test độc lập
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report\n")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

# Vẽ biểu đồ kết quả
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

metrics_data = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [acc, precision, recall, f1]
})

sns.barplot(x='Metric', y='Value', data=metrics_data, ax=axes[0], palette='viridis')
axes[0].set_ylim(0, 1.15)
axes[0].set_title('Đánh Giá Mô Hình LSTM Bằng 4 Độ Đo', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Giá Trị (0.0 - 1.0)')

for p in axes[0].patches:
    axes[0].annotate(
        f"{p.get_height() * 100:.2f}%",
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold'
    )

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
    xticklabels=['Low (0)', 'Medium (1)', 'High (2)'],
    yticklabels=['Low (0)', 'Medium (1)', 'High (2)']
)
axes[1].set_title('Ma Trận Nhầm Lẫn - LSTM (Confusion Matrix)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Giá Trị Dự Đoán (Predicted)')
axes[1].set_ylabel('Giá Trị Thực Tế (Actual)')

plt.tight_layout()
plt.savefig('reports/evaluation_results_lstm.png', dpi=300)
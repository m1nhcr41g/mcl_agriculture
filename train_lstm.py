import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix)

# Đọc dữ liệu dạng chuỗi 3D
data = np.load("data_sequence_for_lstm.npz")
X_train = data["x_train"]
X_test = data["x_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print(f"X_train Shape: {X_train.shape}")
print(f"X_test Shape : {X_test.shape}")

# Mã hóa nhãn Target sang dạng One-Hot Encoding cho Multi-class Classification
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

time_steps = X_train.shape[1]
num_features = X_train.shape[2]

# Xây dựng kiến trúc mô hình LSTM
model = Sequential([
    LSTM(64, input_shape=(time_steps, num_features), return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("TÓM TẮT KIẾN TRÚC MÔ HÌNH LSTM")
model.summary()

# Tính toán Trọng số Lớp (Class Weights) xử lý mất cân bằng
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
print("\nTrọng số lớp (Class Weights):", class_weight_dict)

# Cấu hình Callbacks (EarlyStopping & ModelCheckpoint)
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath="model_lstm.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# Huấn luyện mô hình
EPOCHS = 60
BATCH_SIZE = 32

history = model.fit(
    X_train, y_train_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1
)

# Dự đoán và Tính toán các độ đo Đánh giá
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Tính toán các chỉ số Accuracy, Precision, Recall, F1-Score
acc = accuracy_score(y_test, y_pred)
prec_macro = precision_score(y_test, y_pred, average="macro")
rec_macro = recall_score(y_test, y_pred, average="macro")
f1_macro = f1_score(y_test, y_pred, average="macro")

target_names = ["Low (0)", "Medium (1)", "High (2)"]

print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH LSTM")
print(f"Accuracy Score           : {acc:.4f}")
print(f"Precision (Macro Average): {prec_macro:.4f}")
print(f"Recall (Macro Average)   : {rec_macro:.4f}")
print(f"F1-Score (Macro Average) : {f1_macro:.4f}")
print("\nBẢNG BÁO CÁO CHI TIẾT (CLASSIFICATION REPORT):")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Cột 4 Độ đo
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
values = [acc, prec_macro, rec_macro, f1_macro]
colors = ["#483D8B", "#2E8B57", "#3CB371", "#7CB342"]

bars = axes[0].bar(metrics, values, color=colors, width=0.6)
axes[0].set_title("Đánh Giá Mô Hình Bằng 4 Độ Đo", fontsize=14, fontweight="bold", pad=15)
axes[0].set_ylabel("Giá Trị (0.0 - 1.0)", fontsize=11)
axes[0].set_xlabel("Metric", fontsize=11)
axes[0].set_ylim(0, 1.1)

for bar in bars:
    height = bar.get_height()
    axes[0].text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"{height * 100:.2f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

# Ma trận nhầm lẫn 
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=axes[1],
    xticklabels=target_names,
    yticklabels=target_names
)
axes[1].set_title("Ma Trận Nhầm Lẫn (Confusion Matrix)", fontsize=14, fontweight="bold", pad=15)
axes[1].set_xlabel("Giá Trị Dự Đoán (Predicted)", fontsize=11)
axes[1].set_ylabel("Giá Trị Thực Tế (Actual)", fontsize=11)

plt.tight_layout()
plt.savefig("evaluation_results_lstm.png", dpi=300)

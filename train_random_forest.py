import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
train_df = pd.read_csv('data_train_fe.csv')
test_df = pd.read_csv('data_test_fe.csv')
X_train = train_df.drop(columns=['Target'])
y_train = train_df['Target']
X_test = test_df.drop(columns=['Target'])
y_test = test_df['Target']
rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
metrics_data = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [acc, prec, rec, f1]
})
sns.barplot(x='Metric', y='Value', data=metrics_data, ax=axes[0], palette='viridis')
axes[0].set_ylim(0, 1.15)
axes[0].set_title('Đánh Giá Mô Hình Bằng 4 Độ Đo', fontsize=14, fontweight='bold')
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
axes[1].set_title('Ma Trận Nhầm Lẫn (Confusion Matrix)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Giá Trị Dự Đoán (Predicted)')
axes[1].set_ylabel('Giá Trị Thực Tế (Actual)')
plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=300)
joblib.dump(rf_model, 'model_random_forest.pkl')
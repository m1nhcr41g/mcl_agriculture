import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

train_df = pd.read_csv("data/data_train_fe.csv")
test_df = pd.read_csv("data/data_test_fe.csv")
X_train = train_df.drop(columns=["Target"])
y_train = train_df["Target"]
X_test = test_df.drop(columns=["Target"])
y_test = test_df["Target"]
clf = LGBMClassifier(
    objective="multiclass",
    num_class=3,
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="multi_logloss"
)
preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, average="macro", zero_division=0)
rec = recall_score(y_test, preds, average="macro", zero_division=0)
f1 = f1_score(y_test, preds, average="macro", zero_division=0)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}\n")
print(classification_report(y_test, preds, target_names=["Low", "Medium", "High"], zero_division=0))
cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
metrics_df = pd.DataFrame({'Metric': ["Accuracy", "Precision", "Recall", "F1-Score"], 'Score': [acc, prec, rec, f1]})
sns.barplot(x='Metric', y='Score', data=metrics_df, ax=ax[0], hue='Metric', palette="Blues_d", legend=False)
ax[0].set_ylim(0, 1.15)
ax[0].set_title("LightGBM Metrics")
for p, score in zip(ax[0].patches, [acc, prec, rec, f1]):
    ax[0].annotate(f"{score*100:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.02), ha='center')
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"], ax=ax[1])
ax[1].set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/evaluation_results_lightgbm.png", dpi=300)
plt.show()
joblib.dump(clf, "models/model_lightgbm.pkl")
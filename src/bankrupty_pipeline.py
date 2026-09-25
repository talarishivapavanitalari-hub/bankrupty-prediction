"""
Corporate Bankruptcy Prediction
Decision Tree vs Boosting comparison
Dataset: Taiwanese Bankruptcy Prediction (UCI/Kaggle)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
df = pd.read_csv("data/bankrupty_data.csv")
print("Shape:", df.shape)
print(df["Bankrupt?"].value_counts())

# 2. Split features and target
X = df.drop("Bankrupt?", axis=1)
y = df["Bankrupt?"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Baseline model — Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)

# 4. Boosting model — Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_test)

# 5. Compare results
def evaluate(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

evaluate("Decision Tree", y_test, dt_preds)
evaluate("Gradient Boosting", y_test, gb_preds)

# 6. Feature importance (Decision Tree)
importances = pd.Series(dt_model.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=False).head(10)
print("\nTop 10 important features (Decision Tree):")
print(top10)

plt.figure(figsize=(8, 5))
top10.plot(kind="barh")
plt.title("Top 10 Important Features - Decision Tree")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/feature_importance.png")
print("\nSaved: reports/feature_importance.png")

# 7. Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(confusion_matrix(y_test, dt_preds), annot=True, fmt="d", ax=axes[0], cmap="Blues")
axes[0].set_title("Decision Tree")
sns.heatmap(confusion_matrix(y_test, gb_preds), annot=True, fmt="d", ax=axes[1], cmap="Greens")
axes[1].set_title("Gradient Boosting")
plt.tight_layout()
plt.savefig("reports/confusion_matrices.png")
print("Saved: reports/confusion_matrices.png")
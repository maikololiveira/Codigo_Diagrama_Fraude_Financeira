import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# 1. Load the dataset
df = pd.read_csv("transactions.csv")

# 2. Independent variables (X) and target (y)
X = df.drop("fraud", axis=1)
y = df["fraud"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Class balancing (fraud cases are usually <5%)
classes = [0, 1]
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
weights = dict(zip(classes, class_weights))

# 5. Train the model
model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight=weights
)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Classification report
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# 8. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"],
)
plt.title("Confusion Matrix - Fraud Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 9. Feature importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance in the Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


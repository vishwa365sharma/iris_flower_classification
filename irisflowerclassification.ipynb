# Import Libraries

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pickle

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["Species"] = iris.target


# Data Exploration

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistics")
print(df.describe())

print("\nTarget Count")
print(df["Species"].value_counts())

# ============================================
# Step 4: Data Visualization
# ============================================

df.hist(figsize=(10,8))
plt.tight_layout()
plt.show()

# ============================================
# Step 5: Split Features and Target
# ============================================

X = df.drop("Species", axis=1)
y = df["Species"]

# ============================================
# Step 6: Train Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================
# Step 7: Logistic Regression
# ============================================

lr = LogisticRegression(max_iter=200)

lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test, pred_lr)

print("\n========== Logistic Regression ==========")
print("Accuracy :", acc_lr)
print(confusion_matrix(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

# ============================================
# Step 8: KNN
# ============================================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, pred_knn)

print("\n========== KNN ==========")
print("Accuracy :", acc_knn)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))


# Decision Tree

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)

acc_dt = accuracy_score(y_test, pred_dt)

print("\n========== Decision Tree ==========")
print("Accuracy :", acc_dt)
print(confusion_matrix(y_test, pred_dt))
print(classification_report(y_test, pred_dt))


# Accuracy Comparison

print("\n==============================")
print("Model Accuracy Comparison")
print("==============================")

print("Logistic Regression :", round(acc_lr*100,2), "%")
print("KNN                 :", round(acc_knn*100,2), "%")
print("Decision Tree       :", round(acc_dt*100,2), "%")


# Best Model Selection

models = {
    "Logistic Regression": acc_lr,
    "KNN": acc_knn,
    "Decision Tree": acc_dt
}

best_model = max(models, key=models.get)

print("\nBest Model :", best_model)

if best_model == "Logistic Regression":
    model = lr

elif best_model == "KNN":
    model = knn

else:
    model = dt



pickle.dump(model, open("best_model.pkl", "wb"))

print("\nModel Saved Successfully!")


# Sample Prediction

sample = [[9.1, 6.1, 4.7, 9.8]]

prediction = model.predict(sample)

flower = iris.target_names[prediction[0]]

print("\nSample Prediction :", flower)

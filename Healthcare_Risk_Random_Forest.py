import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split,
                                     GridSearchCV,
                                     cross_val_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             precision_score,
                             recall_score,
                             f1_score,
                             classification_report,
                             ConfusionMatrixDisplay)
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("C:\\Users\\ANANGSHA\\Downloads\\healthcare_random_forest_dataset_7000_complete.csv")
print(df.head())
print(df.tail())
print(df.dtypes)
print(df.columns)
df.drop("Patient_ID", axis=1, inplace=True)
print(df.isnull().sum())
# ============================================================
# Label Encoding
# ============================================================
label_encoder = LabelEncoder()
df["Disease_Risk"] = label_encoder.fit_transform(df["Disease_Risk"])
df = pd.get_dummies(df, drop_first=True)
# ============================================================
# Features and Target
# ============================================================
x = df.drop("Disease_Risk",axis=1)
y = df["Disease_Risk"]
print(df["Disease_Risk"].unique())

# ============================================================
# Splitting Train and Test datas
# ============================================================
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=70, stratify=y)

# ============================================================
# Random Forest Model Creating and Model Training
# ============================================================
rf = RandomForestClassifier(random_state=70)
rf.fit(x_train, y_train)

#-------------------------------------------------
#Hyperparameter Tuning
#-------------------------------------------------
param_grid = {
    "n_estimators":[100, 200],
    "max_depth":[10, None],
    "min_samples_split":[2, 5]
}

#-------------------------------------------------
#Dataset Shape
#-------------------------------------------------
print("Dataset Shape:", df.shape)
print("Feature Matrix Shape:", x.shape)
print("Training Shape:", x_train.shape)

#-------------------------------------------------
#Grid Search CV
#-------------------------------------------------
grid_search = GridSearchCV(
    estimator = rf,
    param_grid=param_grid,
    cv=3,
    scoring="f1_weighted",
    n_jobs=1,
    verbose=2
)
grid_search.fit(x_train, y_train)

#-------------------------------------------------
#Best Model
#-------------------------------------------------
best_rf = grid_search.best_estimator_
print("\nBest Parameters:")
print(grid_search.best_params_)

#-------------------------------------------------
#Predictions
#-------------------------------------------------
y_pred = best_rf.predict(x_test)

#-------------------------------------------------
#Evaluation Metrics
#-------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Accuracy =",accuracy)
print("Precision =",precision)
print("Recall =",recall)
print("F1 =",f1)

#-------------------------------------------------
#Confusion Matrix
#-------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

fig, ax = plt.subplots(figsize=(8,6))
disp.plot(ax=ax)
plt.title("Confusion Matrix")
plt.show()

# ============================================================
# Feature Importance Visualization
# ============================================================

importance_df = pd.DataFrame({
    "Feature": x.columns,
    "Importance": best_rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# ============================================================
# Top 10 Features
# ============================================================
top_features = importance_df.head(10)

plt.figure(figsize=(8, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Important Features")

plt.gca().invert_yaxis()

plt.show()

# ============================================================
# Cross Validation
# ============================================================
cv_scores = cross_val_score(best_rf,
                           x, y,
                           cv=5,
                           scoring='accuracy')
print("Cross Validation Scores=",cv_scores)

# ============================================================
# Classification Report
# ============================================================
classified_report = classification_report(y_test, y_pred)
print("Classification Report:",classified_report)


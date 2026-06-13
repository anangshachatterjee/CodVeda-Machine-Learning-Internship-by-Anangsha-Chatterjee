import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ( confusion_matrix,
                              precision_score,
                              recall_score,
                              classification_report,
                              roc_curve,
                              roc_auc_score)

df = pd.read_csv("C:\\Users\\ANANGSHA\\Downloads\\churn-bigml-80.csv")
print(df.head())
print(df.tail())
df = pd.get_dummies(df, columns=["State"], drop_first=True)
print(df.dtypes)
print("\nDataset Shape")
print(df.shape)

#-------------------------------------------------
#Check missing values
#-------------------------------------------------
print(df.isnull().sum())

#-------------------------------------------------
#Label Encoding
#-------------------------------------------------
label_encoder = LabelEncoder()
df["International plan"] = label_encoder.fit_transform(df["International plan"])
df["Voice mail plan"] = label_encoder.fit_transform(df["Voice mail plan"])
df["Churn"] = label_encoder.fit_transform(df["Churn"])

#-------------------------------------------------
#Features and Target 
#-------------------------------------------------
x=df.drop("Churn",axis=1)
y=df["Churn"]

#-------------------------------------------------
#Splitting Train and Test datas
#-------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=50, stratify=y)

#-------------------------------------------------
#Data Scaling
#-------------------------------------------------
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#-------------------------------------------------
#Logistic Regression model fitting
#-------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

#-------------------------------------------------
#Prediction of data based on test values
#-------------------------------------------------
y_pred = model.predict(x_test)
print(y_pred[ :10])
y_prob = model.predict_proba(x_test)[:,1]

#-------------------------------------------------
#Compute accuracy score
#-------------------------------------------------
def compute_accuracy(y_test, y_pred):
    y_test = np.array(y_test)
    correct=0
    for i in range(len(y_test)):
        if y_test[i]==y_pred[i]:
             correct+=1
    accuracy=correct/(len(y_test))

    return accuracy

accuracy_score = compute_accuracy(y_test, y_pred)
print("Accuracy=",accuracy_score)

#-------------------------------------------------
#Compute confusion matrix
#-------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix=", cm)

#-------------------------------------------------
#Compute precision score
#-------------------------------------------------
precision = precision_score(y_test, y_pred, zero_division=0)
print("Precision Score=", precision)

#-------------------------------------------------
#Compute recall score
#-------------------------------------------------
recall = recall_score(y_test, y_pred, zero_division=0)
print("Recall Score=", recall)

#-------------------------------------------------
#Compute classification report
#-------------------------------------------------
classified_report = classification_report(y_test, y_pred)
print("\nClassification Report=", classified_report)

#-------------------------------------------------
#Compute ROC-AUC score
#-------------------------------------------------
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC score=", roc_auc)

#-------------------------------------------------
#ROC Curve
#-------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8,8))
plt.plot(fpr,tpr,label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

#-------------------------------------------------
#Model Coefficients
#-------------------------------------------------
intercept = model.intercept_
print("Intercept=", intercept)

coefs = model.coef_
print("Co-efficient=", coefs[ :10])


#-------------------------------------------------
#Odds ratio
#-------------------------------------------------
odds_ratio = np.exp(coefs)
print("Odds-ratio=", odds_ratio)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import(accuracy_score,
                            precision_score,
                            recall_score,
                            f1_score,
                            confusion_matrix,
                            classification_report)

df = pd.read_excel("C:\\Users\\ANANGSHA\\Documents\\cybersecurity_intrusion_detection_dataset.csv.xlsx")
print(df.head())
print(df.tail())
print(df.dtypes)

#-------------------------------------------------
#Features and Target 
#-------------------------------------------------
x = df[['Packet_Size', 'Request_Frequency']]
y = df["Attack_Type"]
print(df['Attack_Type'].unique())

#-------------------------------------------------
#Splitting Test and Train datas
#-------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=50, stratify=y)

#-------------------------------------------------
#Feature Scaling
#-------------------------------------------------
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#-------------------------------------------------
#SVM models creating, fitting and prediction
#-------------------------------------------------
linear_svm = SVC(kernel='linear', C=1.0)
rbf_svm = SVC(kernel='rbf', C=1.0, gamma='scale')

linear_svm.fit(x_train_scaled, y_train)
rbf_svm.fit(x_train_scaled, y_train)

linear_pred = linear_svm.predict(x_test_scaled)
rbf_pred = rbf_svm.predict(x_test_scaled)

#-------------------------------------------------
#Metrics Evaluation for Linear SVM
#-------------------------------------------------
accuracy_linear = accuracy_score(y_test, linear_pred)
precision_linear = precision_score(y_test, linear_pred, average='weighted')
recall_linear = recall_score(y_test, linear_pred, average='weighted')
f1_linear = f1_score(y_test, linear_pred, average='weighted')

print("Accuracy:",accuracy_linear)
print("Precision:",precision_linear)
print("Recall:",recall_linear)
print("F1:",f1_linear)

#-------------------------------------------------
#Confusion Matrix and Classification Report for Linear SVM
#-------------------------------------------------
cm_linear = confusion_matrix(y_test, linear_pred)
print("Confusion Matrix:",cm_linear)

classified_report_linear = classification_report(y_test, linear_pred)
print("Classification Report:",classified_report_linear)

#-------------------------------------------------
#Metrics Evaluation for RBF SVM
#-------------------------------------------------
accuracy_rbf = accuracy_score(y_test, rbf_pred)
precision_rbf = precision_score(y_test, rbf_pred, average='weighted')
recall_rbf = recall_score(y_test, rbf_pred, average='weighted')
f1_rbf = f1_score(y_test, rbf_pred, average='weighted')

print("Accuracy:",accuracy_rbf)
print("Precision:",precision_rbf)
print("Recall:",recall_rbf)
print("F1:",f1_rbf)

#-------------------------------------------------
#Confusion Matrix and Classification Report for RBF SVM
#-------------------------------------------------
cm_rbf = confusion_matrix(y_test, rbf_pred)
print("Confusion Matrix:",cm_rbf)

classified_report_rbf = classification_report(y_test, rbf_pred)
print("Classification Report:",classified_report_rbf)


#-------------------------------------------------
#SVM Performance comparision
#-------------------------------------------------
results = pd.DataFrame({
    "Kernel": ["Linear", "RBF"],
    "Accuracy": [
        accuracy_score(y_test, linear_pred),
        accuracy_score(y_test, rbf_pred)
    ],
    "Precision": [
        precision_score(y_test, linear_pred, average='weighted'),
        precision_score(y_test, rbf_pred, average='weighted')
    ],
    "Recall": [
        recall_score(y_test, linear_pred, average='weighted'),
        recall_score(y_test, rbf_pred, average='weighted')
    ],
    "F1 Score": [
        f1_score(y_test, linear_pred, average='weighted'),
        f1_score(y_test, rbf_pred, average='weighted')
    ]
})
print("Performance Comparision:",results)

#-------------------------------------------------
#Visualization for Linear and RBF SVMs
#-------------------------------------------------
def plot_boundary(model, x_train_scaled, y_train, title):
    
    x_min, x_max = x_train_scaled[:,0].min()-1, x_train_scaled[:,0].max()+1
    y_min, y_max = x_train_scaled[:,1].min()-1, x_train_scaled[:,1].max()+1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Convert labels to integers
    Z = pd.factorize(Z)[0]
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8,6))

    plt.contourf(xx, yy, Z, alpha=0.3)

    plt.scatter(
        x_train_scaled[:,0],
        x_train_scaled[:,1],
        c=pd.factorize(y_train)[0],
        edgecolors='k'
    )

    plt.xlabel("Packet_Size")
    plt.ylabel("Request_Frequency")
    plt.title(title)

    plt.show()

plot_boundary(linear_svm, x_train_scaled, y_train, "Linear SVM Decision Boundary")
plot_boundary(rbf_svm, x_train_scaled, y_train, "RBF SVM Decision Boundary")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv("C:\\Users\\ANANGSHA\\Downloads\\student_performance_knn_dataset.csv")
print(df.head())
print(df.tail())
print(df.dtypes)

#-------------------------------------------------
#Features and Target 
#-------------------------------------------------
x=df[['study_hours','attendance_percent']]
x = x.astype(float)
y=df['performance_class']
print(df['performance_class'].unique())

#-------------------------------------------------
#Splitting Train and Test datas
#-------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=40)

#-------------------------------------------------
#KNN model fitting
#-------------------------------------------------
n_neighbors = int(input("Enter K values 1,3,5,7 or 9:"))
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(x_train, y_train)

#-------------------------------------------------
#Prediction of data based on test values
#-------------------------------------------------
print("\nPredicted Values")
y_pred = knn.predict(x_test)
print(y_pred)

print("\nActual Values")
y_test = np.array(y_test)
print(y_test)

#-------------------------------------------------
#Compute accuracy score
#-------------------------------------------------
def compute_accuracy(y_test, y_pred):
    correct=0
    for i in range(len(y_test)):
        if y_test[i]==y_pred[i]:
             correct+=1
    accuracy=correct/(len(y_test))

    return accuracy

accuracy_score = compute_accuracy(y_test, y_pred)
print("Accuracy=",accuracy_score)

#-------------------------------------------------
#Create a confusion matrix
#-------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:",cm)

#-------------------------------------------------
#Plotting
#-------------------------------------------------
plt.figure(figsize=(8,8))
for result in np.unique(y):
    plt.scatter(
        df[df['performance_class'] == result]['study_hours'],
        df[df['performance_class'] == result]['attendance_percent'],
        label=f"performance_class {result}"
        )
plt.xlabel("study_hours")
plt.ylabel("attendance_percent")
plt.title("KNN Classification on Students' Performance")
plt.legend()
plt.show()

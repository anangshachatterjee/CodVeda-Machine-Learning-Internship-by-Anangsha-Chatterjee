import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

df = pd.read_csv("C:\\Users\\ANANGSHA\\Downloads\\1) iris.csv")
print(df.head())
print(df.tail())
print(df.dtypes)
print(df.columns)

#-------------------------------------------------
#Features and Target 
#-------------------------------------------------
x = df[['sepal_length','sepal_width','petal_length','petal_width']]
y = df['species']
print(df['species'].unique())

#-------------------------------------------------
#Splitting Train and Test datas
#-------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 60)

#-------------------------------------------------
#Classifier Decision Tree (pruned to prevent overfitting) model fitting
#-------------------------------------------------
model = DecisionTreeClassifier(criterion = 'gini',
                               max_depth = 4,
                               min_samples_split = 3,
                               min_samples_leaf = 2,
                               max_leaf_nodes = 12,
                               random_state = 60)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(y_pred[ :10])

#-------------------------------------------------
#Compute Accuracy score
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
#Compute F1 score
#-------------------------------------------------
f1 = f1_score(y_test, y_pred, average = 'weighted')
print("F1 score=",f1)

#-------------------------------------------------
#Visualising the tree
#-------------------------------------------------
plt.figure(figsize=(8,4))
tree.plot_tree(model,
               feature_names = x.columns,
               class_names = model.classes_,
               filled=True)
plt.show()

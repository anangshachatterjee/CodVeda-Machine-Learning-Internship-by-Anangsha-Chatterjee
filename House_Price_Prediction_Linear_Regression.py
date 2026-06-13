import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

columns = ["municipal_tax(in %)",
               "industrial_area(in %)",
               "property_tax (in %)",
               "has_swimming_pool",
               "Nearest_metro(in km)",
               "Value_Added_Tax(in %)",
               "Land_area_used (in hectares)",
               "distance_to_school(in km)",
               "access_to_highway",
               "house_sizes (in sq. ft)",
               "labour_charges(in lakhs)",
               "furniture_and_interior_charges(in Lakhs)",
               "GST (in %)",
               "house_price (in Lakhs)"]
df = pd.read_csv("C:\\Users\\ANANGSHA\\Downloads\\4) house Prediction Data Set.csv",sep=r'\s+',
                 header=None,names=columns)

df.drop(columns=["municipal_tax(in %)","industrial_area(in %)","has_swimming_pool",
                 "Nearest_metro(in km)","distance_to_school(in km)","access_to_highway",
                 "furniture_and_interior_charges(in Lakhs)"],inplace=True)

print(df.head())
print(df.tail())

#-------------------------------------------------
#Features and Target (x and y variables)(House Size vs House Prices)
#-------------------------------------------------
x = df[["house_sizes (in sq. ft)"]]
y = df["house_price (in Lakhs)"]

#-------------------------------------------------
#Splitting Train and Test datas
#-------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.3, random_state=50)

#-------------------------------------------------
#Data Scaling
#-------------------------------------------------
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#-------------------------------------------------
#Linear Regression Model Fitting
#-------------------------------------------------

model = LinearRegression()
model.fit(x_train, y_train)

#-------------------------------------------------
#Prediction of prices based on size
#-------------------------------------------------
y_pred = model.predict(x_test)
print(y_pred[ :10])

#-------------------------------------------------
#Co-efficients
#-------------------------------------------------

slope = model.coef_[0]
intercept = model.intercept_

#-------------------------------------------------
#Compute Sum-squared error
#-------------------------------------------------
def compute_sse (y_test, y_pred):
    error = y_test - y_pred
    errors = np.array(error)
    sse = np.sum(error ** 2)
    return sse

#-------------------------------------------------
#Compute Root mean-squared error
#-------------------------------------------------
def compute_rmse (y_test, y_pred):
    mse = np.mean((y_test - y_pred)**2)
    rmse = np.sqrt(mse)
    return rmse

#-------------------------------------------------
#Compute R2 value
#-------------------------------------------------
def compute_r2(y_test, y_pred):
    ss_residual = np.sum((y_test - y_pred)**2)
    mean_y = np.mean(y_test)
    ss_total = np.sum((y_test - mean_y)**2)
    r2 = 1 - (ss_residual/ss_total)
    return r2

#-------------------------------------------------
#Evaluation of metrics
#-------------------------------------------------
sse = compute_sse (y_test, y_pred)
rmse = compute_rmse (y_test, y_pred)
r2 = compute_r2(y_test, y_pred)

print("\nPerformance Metrics:")
print("Sum-squared error =",sse)
print("Root mean-squared error =",rmse)
print("R2 =",r2)

#-------------------------------------------------
#Graph
#-------------------------------------------------
plt.figure(figsize = (8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
    

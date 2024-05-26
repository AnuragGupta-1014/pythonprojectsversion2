                 # STOCK PRICE PREDICTION
import os
import pandas as pd 
import numpy as np

import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns

              
# Data Collection and Preprocessing

data_path = 'Stock_Price_Prediction.csv'
if os.path.exists(data_path):
    Data = pd.read_csv(data_path)
    print(Data.head())
    print(Data.tail())
    print(Data.loc[[1, 2]])
    print(Data.index)
    print("Shape of Datasheet:", Data.shape)
    print("Columns of Datasheet:", Data.columns)
else:
    print(f"File {data_path} not found.")
                                 
print(Data.isna()) 
print(Data.isna().sum())
print(Data.head().dropna())
print(Data.tail().sort_values)

# Data Analysis

# Line plot
Data['High'] = pd.to_numeric(Data['High'])
Data['Low'] = pd.to_numeric(Data['Low'])
Data.head(144).plot(kind="line", x="High", y="Low")
plt.xlabel("High")
plt.ylabel("Low")
plt.title("High vs Low")
plt.show()


# Pie plot
df = Data.head(14)["Volume"].plot(kind= "pie")
plt.show()  

# Scatter Plot
ploting = Data.plot(kind="scatter",x= "High", y= "Low") 
plt.title("Stock price Prediction")                             
plt.xlabel("High")                              
plt.ylabel("Low")
plt.show()

# Exploratory Data Analysis (EDA) 
print("--------------------------------------------------------------------------")
print(Data["Adj Close"].isnull().sum())                                       # Before Filling
Median = Data['Adj Close'].std()
print("Median of  columns = ", Median,"\n")                  # filling missing value with avg   
Data['Adj Close'] = Data['Close'].fillna(Median)
print(Data["Adj Close"].isnull().sum())                             # After Filling
print("--------------------------------------------------------------------------")
           
           
           
# Outlier
sns.boxenplot(x =Data["Volume"])
plt.show()

sns.boxplot(y = Data["Close"])
plt.show()

# Feature Engineering

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression   
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization
scaler_standard = StandardScaler()
Data_standardized = scaler_standard.fit_transform(Data[['Open', 'High', 'Close']])
Data_standardized = pd.DataFrame(Data_standardized, columns=['Open', 'High', 'Close'])

# Normalization
scaler_minmax = MinMaxScaler()
Data_normalized = scaler_minmax.fit_transform(Data[['Open', 'High', 'Close']])
Data_normalized = pd.DataFrame(Data_normalized, columns=['Open', 'High', 'Close'])

# Plotting the standardized and normalized data

plt.subplot(2, 3, 1)
sns.histplot(Data_normalized['Open'])
plt.title('Normalized Data')

plt.subplot(2, 3, 5)
sns.histplot(Data_standardized['High'])
plt.title('Standardized Data')

plt.subplot(2, 3, 3)
sns.histplot(Data_normalized['Close'])
plt.title('Normalized Data')


plt.tight_layout()
plt.show()

# Model Building and It's Accuracy

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Selecting features and target variable
X = Data[['Open', 'High', 'Low', 'Volume']]
y = Data['Adj Close']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building a Random Forest regression model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
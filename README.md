# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KISHORE M
RegisterNumber:  212223040100

```

```python

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE) : ", mse)

r2=metrics.r2_score(y_test,y_pred)
print("R2 : ",r2)

dt.predict([[5,6]])

```

## Output:

### DATASET:

![image](https://github.com/user-attachments/assets/89b4d782-beea-4644-bb8b-c69b58ca19da)

![image](https://github.com/user-attachments/assets/1d3f425d-ef90-47ef-8a97-2a1033878a3c)

![image](https://github.com/user-attachments/assets/8d2db0af-3e82-4b59-9c7f-466b962e290e)


### LABELLED DATASET:

![image](https://github.com/user-attachments/assets/c180d9f6-dc20-48c3-98f3-efbccae1c648)

### MEAN SQUARED ERROR:

![image](https://github.com/user-attachments/assets/1d5e1976-d1f3-49a2-8ab9-3650076c14c6)


### R2 VALUE:
![image](https://github.com/user-attachments/assets/079ac528-54d1-4c94-a33a-bba053473fe0)

### PREDICTION VALUE:
![image](https://github.com/user-attachments/assets/fad2d24a-976f-4e74-b90a-829059022964)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

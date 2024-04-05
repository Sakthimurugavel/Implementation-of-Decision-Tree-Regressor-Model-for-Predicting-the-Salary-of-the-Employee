# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the packages that helps to implement Decision Tree.
2. Download and upload required csv file or dataset for predecting Employee Churn.
3. Initialize variables with required features.
4. And implement Decision tree classifier to predict Employee Churn. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SAKTHIVEL M
RegisterNumber:  212222240088
*/
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

from sklearn.tree import DecisionTreeRegressor, plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plot_tree(dt,feature_names=x.columns,class_names=['Salary'], filled=True)
plt.show()

```

## Output:
### 1. Head:
![out 1 ex 7](https://github.com/Sakthimurugavel/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707246/0d77377e-0039-431b-af7f-e9ebecd75276)

### 2. Mean square error:
![out 2 ex 7](https://github.com/Sakthimurugavel/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707246/6608e830-4e3c-46d7-9ee2-c283584d07f2)

### 3. Testing of Model:
![out 3 ex 7](https://github.com/Sakthimurugavel/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707246/46ff1dbf-d68b-47e8-a0fe-d11bbc4df404)

### 4. Decision Tree:
![out 4 ex 7](https://github.com/Sakthimurugavel/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118707246/57b97cb3-450c-4dd8-b5f4-58a654557d7d)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

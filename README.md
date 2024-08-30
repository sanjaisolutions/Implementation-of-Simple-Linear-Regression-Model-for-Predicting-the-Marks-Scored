# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages.
2.Assigning hours To X and Scores to Y.
3.Plot the scatter plot.
4.Use mse,rmse,mae formula to find the values.

## Program:
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SANJAI.R
RegisterNumber:  212223040180

```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('/content/student_scores.csv')
print('df.head')

df.head()

print("df.tail")
df.tail()

X=df.iloc[:,:-1].values
print("Array of X")
X

Y=df.iloc[:,1].values
print("Array of Y")
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

print("Values of Y prediction")
Y_pred

print("Values of Y test")
Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Training Set Graph")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("scores")
print("Test Set Graph")
plt.show()

print("Values of MSE, MAE and RMSE")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![Screenshot 2024-08-30 105747](https://github.com/user-attachments/assets/dd254b69-9c24-4dfd-8eb3-5f5baabb70b5)

![Screenshot 2024-08-30 105752](https://github.com/user-attachments/assets/05a4ae1f-45a4-4923-abb4-b16236ea8fdf)

![Screenshot 2024-08-30 105757](https://github.com/user-attachments/assets/72fc137e-51fc-4ba0-89b6-3a38e74e8938)

![Screenshot 2024-08-30 105801](https://github.com/user-attachments/assets/ac52f954-c9b3-49fc-ad76-7516f355a5fe)


![Screenshot 2024-08-30 105805](https://github.com/user-attachments/assets/3317f2ac-4787-47ea-bf81-c699b74ddb15)

![Screenshot 2024-08-30 105810](https://github.com/user-attachments/assets/b8d63b56-c964-4b63-8482-d302cbf1d3f3)

![Screenshot 2024-08-30 105814](https://github.com/user-attachments/assets/0ab9c8ed-2ad2-4604-a129-c6e02416a476)

![Screenshot 2024-08-30 105818](https://github.com/user-attachments/assets/2139f844-a5fb-4f48-bac0-4f4ebb586d66)

![Screenshot 2024-08-30 105821](https://github.com/user-attachments/assets/7fd4a50e-9236-4d5c-9798-bc1c7c237911)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

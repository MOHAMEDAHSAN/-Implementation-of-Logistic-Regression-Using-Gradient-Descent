# EXP5- Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters: Set initial values for the weights (w) and bias (b).
2. Compute Predictions: Calculate the predicted probabilities using the logistic function.
3. Compute Gradient: Compute the gradient of the loss function with respect to w and b.
4. Update Parameters: Update the weights and bias using the gradient descent update rule. Repeat steps 2-4 until convergence or a maximum number of iterations is reached.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S MOHAMED AHSAN
RegisterNumber:  212223240089
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
```
data=pd.read_csv("Placement_Data.csv")
data.head()

data=data.drop('sl_no',axis=1)
data=data.drop('salary',axis=1)
data.head()

data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data.dtypes

data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data

X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values
Y

theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1 /(1+np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:

![Screenshot 2024-05-04 192344](https://github.com/MOHAMEDAHSAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331378/fa5c62dc-f7ba-4491-9e17-46c4842ff418)
![Screenshot 2024-05-04 192349](https://github.com/MOHAMEDAHSAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331378/049d38d2-e33e-4f8e-860f-6023c25a5e68)
![Screenshot 2024-05-04 192354](https://github.com/MOHAMEDAHSAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331378/51fb4dcd-9c0d-4201-807c-827a227a8bc8)
![Screenshot 2024-05-04 192359](https://github.com/MOHAMEDAHSAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331378/eb4dfeed-2447-48a1-b923-635ae6c1ccbb)
![Screenshot 2024-05-04 192404](https://github.com/MOHAMEDAHSAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331378/de861a12-2d19-49ae-8e5f-d9b7ed9fa83e)

<BR>

![Screenshot 2024-05-04 192409](https://github.com/MOHAMEDAHSAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331378/83cb3923-fc8e-4d51-99e2-f13265d400b4)
![Screenshot 2024-05-04 192414](https://github.com/MOHAMEDAHSAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331378/b5e1170d-4e5a-45f1-9abb-a880b4b094f5)
![Screenshot 2024-05-04 192427](https://github.com/MOHAMEDAHSAN/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/139331378/ab79dfff-e0ab-48eb-af83-d3df0ed0cbc8)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


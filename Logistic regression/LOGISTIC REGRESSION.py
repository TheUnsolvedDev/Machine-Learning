import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#The code for importing the dataset using sklearn is given below.
#"from sklearn.datasets import load_iris"
data = pd.read_csv("iris.csv")
x_train = data.iloc[:,:4].to_numpy()
non_virginica = [0 for i in range(100)]
virginica = [1 for i in range(50)]
y_train = np.concatenate([non_virginica,virginica])

#Creating the train data and test data.
perm = np.random.permutation(150)
x_train,x_test = x_train[perm][20:],x_train[perm][:20]
y_train,y_test = y_train[perm][20:],y_train[perm][:20]
x_train.shape,x_test.shape,y_train.shape,y_test.shape  
y_train = y_train.reshape(y_train.shape[0],1) 

#Creating the weight(theta) and bias(b) matrix. 

theta = np.zeros((x_train.shape[1],1)) 
b = np.zeros((1,1)) # matrix of the bias.
learning_rate = 0.01 

# This sigmoid function gives the probability of the outcome
def sigmoid(z):  
    return 1/(1+np.exp(-z))

# This is the loss function of logistic regression.
# It is similar to maximum likelihood estimation.
def logistic_loss(y,y_pred):
    return -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

#Gradient Descent
def Gradient_descent(x_train,y_train,theta,b):
    m = len(y_train)  # The size of the training example
    
    for i in range(10000):
        # we calculate H(X) = Q^TX + b.
        linear_model = np.dot(x_train,theta) + b  

        # probability of the output using sigmoid function.
        y_pred = sigmoid(linear_model)    
        
        # The loss function of the predicted output
        loss = logistic_loss(y_train,y_pred) 
        
        # the derivative of the loss function is given below
        dloss = (1/m) * np.matmul(x_train.T,(y_pred-y_train)) 
        db = np.sum(y_pred-y_train)  # derivative of the bias
        
        # Here we update the values of theta and bias to get less error.
        theta = theta - learning_rate * dloss 
        b = b - learning_rate * db

        # this line is only to verify that error is decreasing
        if i % 100 == 0:
            print('losgistic_loss={}'.format(loss))        
    return theta,b                

# predict function below is used to predict the output of the given input.
def predict(test_array,logistic_vals):
    predicted_vals = []
    #linear prediction of new input
    predict_linear_model = np.dot(test_array,logistic_vals[0]) + logistic_vals[1] 
    for i in sigmoid(predict_linear_model):
        if i > 0.5:
            predicted_vals.append(1)
        else:
            predicted_vals.append(0)
    return predicted_vals            

#Here we train our model and test our model with the data that we created. 
#After the training we get the optimum values of weight(theta) and bias. 

# The training model is given below. Here we get the values of theta and bias.
Training_model = Gradient_descent(x_train,y_train,theta,b)  

# using this we are calculating our model
test_model = predict(x_test,Training_model) 

print('Output of test data = {}'.format(test_model))


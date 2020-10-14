# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 19:52:46 2020

@author: Srishti Ganguly
"""

#Task 2: Loading Data and Libraries

import matplotlib.pyplot as plt 
plt.style.use('ggplot')
%matplotlib inline

import numpy as np
import pandas as pd  
import seaborn as sns 
plt.rcParams['figure.figsize'] = (12, 8)
data = pd.read_csv("bike_sharing_data.txt")
data.head()

#Task 3: Visualizing the Data
ax = sns.scatterplot(x="Population", y="Profit", data = data)
ax.set_title("Profit in 10000$ vs City Population")

#Task 4: Compute the Cost
def cost_fnc(X,y,theta):
    m = len(y)
    h_x = X.dot(theta)
    error = (h_x - y)**2
    return 1/(2*m)*np.sum(error)

m = data.Population.values.size
X = np.append(np.ones((m,1)), data.Population.values.reshape(m,1), axis=1)
y = data.Profit.values.reshape(m,1)
theta = np.zeros((2,1))
cost_fnc(X,y,theta)

#Task 5: Gradient Descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    costs = []
    for i in range(iterations):
        h_x = X.dot(theta)
        error = np.dot(X.transpose(), (h_x-y))
        theta -= alpha*(1/m)*error
        costs.append(cost_fnc(X, y, theta))
    return theta, costs

theta, costs = gradient_descent(X, y, theta, alpha = 0.01, iterations = 2000)
print("h(x) = {} + {}x1".format(str(round(theta[0,0],2)), str(round(theta[1,0],2))))

#Task 6: Visualizing the Cost Function

from mpl_toolkits.mplot3d import Axes3D

theta_0 = np.linspace(-10,10,100)
theta_1 = np.linspace(-1,4,100)
cost_value = np.zeros((len(theta_0), len(theta_1)))

for i in range(len(theta_0)):
    for j in range(len(theta_1)):
        t = np.array([theta_0[i], theta_1[j]])
        cost_value[i,j] = cost_fnc(X, y, t)

fig = plt.figure(figsize = (12,8))
ax = fig.gca(projection = '3d')

surf = ax.plot_surface(theta_0, theta_1, cost_value, cmap = 'viridis')
fig.colorbar(surf, shrink = 0.5, aspect = 5)

plt.xlabel("$\Theta_0$")
plt.ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")
ax.view_init(30,330)
plt.show()

#Task 7: Plotting the Convergence
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("$J(\Theta)$")
plt.title("Values of the Cost Function over Iteraions of Gradient Descent")

#Task 8: Training Data with Linear Regression Function
theta = np.squeeze(theta)
sns.scatterplot(x="Population", y="Profit", data = data)

x_values = [x for x in range(5,25)]
y_values = [(x * theta[1] + theta[0]) for x in x_values]
sns.lineplot(x_values, y_values)

plt.xlabel("Population in 1000s")
plt.ylabel("Profit in 10000s")
plt.title("Linear Regression Fit")

#Task 9: Inference using optimized Theta Values
def predict(x, theta):
    h_x = np.dot(theta.transpose(),x)
    return h_x
y_pred1 = predict(np.array([1,4]), theta) * 10000
print("For a population of 40000 people, the model predicts a profit of $"+ str(round(y_pred1,0))
y_pred2 = predict(np.array([1,8.3]), theta) * 10000
print("For a population of 83000 people, the model predicts a profit of $"+ str(round(y_pred2,0)))

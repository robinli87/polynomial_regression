#find polynomial regression through dataset

import math
import cmath
import numpy
import random

N = 5

coefficients = [random.gauss(0, 1)]*N


#train data
X = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
Y = [25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25]

x_norm = []
y_norm = []


#parameterised weights and biases use lowercase letters
def run(xi, c):
    guess = 0
    for i in range(0, N):
        guess += c[i] * (xi **  i)
    return(guess)

def loss(c):
    #compute loss based on current w and b
    error = 0
    for i in range(0, len(X)):
        error = error + ( run(X[i], c) - Y[i] ) ** 2

    return(error)

benchmark = loss(coefficients)
print("benchmark: ", benchmark)
#now we start training
#must iterate through all parameters to calculate differentials and modifiers.
#need a function which sees this parameter then varies it

#define weightReference, biasReference
#define quantities learning rate and dw, db (for differentials)
learning_rate = 10 ** -(1.4 * N)
dw = 0.0000001
epoch = 1

def backpropagation():
    #we expect weightReference to be a list of 1 number, biasReference to be  a list with 1 number

    def modififyWeight(indices):
        upper = []
        lower = []
        for i in range(0, len(coefficients)):
            if i == indices:
                upper.append(coefficients[i] + dw)
                lower.append(coefficients[i] - dw)
            else:
                upper.append(coefficients[i])
                lower.append(coefficients[i])

        gradient = (loss(upper) - loss(lower)) / (2 * dw)
        #print("weight gradient = ", gradient)
        coefficients[indices] -= learning_rate * gradient #* (math.e ** (0.001 / gradient))


    #first cycle
    #make changes to parameters
    for i in range(0, len(coefficients)):
        modififyWeight(i)

backpropagation()
new_loss = loss(coefficients)
print(new_loss)

#print("current weights: ", weights)
#print("current biases: ", biases)
#print("new loss: ", new_loss)
while (new_loss<= benchmark):
    benchmark = new_loss
    epoch += 1

    # we can do something fancy about learning rate


    backpropagation()
    new_loss = loss(coefficients)
    print("new_loss = ", new_loss)

print(coefficients)

while True:
    new_x = float(input("Enter a value of x: "))
    prediction = run(new_x, coefficients)
    print(prediction)

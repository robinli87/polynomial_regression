#find polynomial regression through dataset

import math
import cmath
import numpy
import random

#declarative use of weights (official storage) use full English words
weights = [random.random()]*3
biases = [random.random()]*3

print("initial weights: ", weights)
print("initial biases: ", biases)

#train data
X = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
Y = [25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25]

x_norm = []
y_norm = []
# we try to normalise the dataset to keep things sane
for i in X:
    i = (i - min(X)) / (max(X)-min(X))
    x_norm.append(i)

for i in Y:
    i = (i - min(Y)) / (max(Y)-min(Y))
    y_norm.append(i)


#"activation functions"
def bypass(z):
    return(z)

def exponential(z):
    return(cmath.exp(z))

def ln(z):
    try:
        return(cmath.log(z, math.e))
    except:
        print("ValueError encountered, replacing")
        return(-1000)

#parameterised weights and biases use lowercase letters
def run(xi, w, b):
    #run the net once with 1 input and a specified set of weights and biases
    z1 = w[0] * xi + b[0]
    #this z1 will eventually be turned into an array, e.g. z[1], z[1][1]
    a1 = ln(z1)
    z2 = w[1] * a1 + b[1]
    a2 = exponential(z2)
    y = w[2] * a2 - b[2]
    return(y.real)

def loss(w, b):
    #compute loss based on current w and b
    error = 0
    for i in range(0, len(X)):
        error = error + ( run(X[i], w, b) - Y[i] ) ** 2

    return(error)

benchmark = loss(weights, biases)
print("benchmark: ", benchmark)
#now we start training
#must iterate through all parameters to calculate differentials and modifiers.
#need a function which sees this parameter then varies it

#define weightReference, biasReference
#define quantities learning rate and dw, db (for differentials)
learning_rate = 0.001
dw = 0.0000001
db = 0.0000001

def backpropagation():
    #we expect weightReference to be a list of 1 number, biasReference to be  a list with 1 number

    def modififyWeight(indices):
        upper = []
        lower = []
        for i in range(0, len(weights)):
            if i == indices:
                upper.append(weights[i] + dw)
                lower.append(weights[i] - dw)
            else:
                upper.append(weights[i])
                lower.append(weights[i])

        gradient = (loss(upper, biases) - loss(lower, biases)) / (2 * dw)
        #print("weight gradient = ", gradient)
        weights[indices] -= learning_rate * gradient

    def modifyBias(indices):
        upper = []
        lower = []
        for i in range(0, len(biases)):
            if i == indices:
                upper.append(biases[i] + db)
                lower.append(biases[i] - db)
            else:
                upper.append(biases[i])
                lower.append(biases[i])
        gradient = (loss(weights, upper) - loss(weights, lower)) / (2 * dw)
        #print("bias gradient = ", gradient)
        biases[indices] -= learning_rate * gradient

    #first cycle
    #make changes to parameters
    for i in range(0, len(weights)):
        modififyWeight(i)

    for i in range(0, len(biases)):
        modifyBias(i)

backpropagation()
new_loss = loss(weights, biases)
epoch = 1
#print("current weights: ", weights)
#print("current biases: ", biases)
#print("new loss: ", new_loss)
while (abs(new_loss) < abs(benchmark)) and (epoch < 100000):
    benchmark = new_loss
    epoch += 1
    backpropagation()
    new_loss = loss(weights, biases)
    print("new_loss = ", new_loss)

print(weights)
print(biases)

while True:
    new_x = float(input("Enter a value of x: "))
    prediction = run(new_x, weights, biases)
    print(prediction)

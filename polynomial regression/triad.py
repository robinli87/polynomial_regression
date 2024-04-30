#triad structure.py
import random

w = []

structure = [1, 3, 3, 1]   #first number is input layer, last number is output layer, all middle numbers are hidden layers.

for l in range(0, len(structure) - 1):
    #generate dyad of connections. dimensions = this x next

    rows = structure[l]
    cols = structure[l+1]
    this_dyad = []
    for j in range(0, rows):
        this_vector = []
        for k in range(0, cols):
            this_vector.append(random.random())
        this_dyad.append(this_vector)

    w.append(this_dyad)

print(w)
print("=====================================================")

z = []

for l in range(0, len(structure)):
    #generate col vector
    this_col = []
    for k in range(0, structure[l]):
        this_col.append(random.random())
    z.append(this_col)

print(z)

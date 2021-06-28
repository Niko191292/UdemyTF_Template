import numpy as np


my_list = []

for i in range(10):
    my_list.append(i)

print(my_list)


# List Comprehension
my_list2 = [i for i in range(10)]
print(my_list2)

my_list3 = [i**2 for i in range(10)]
print(my_list3)

# Multi-dim List
M = [[1, 2],
     [3, 4]]
print(M)

NumRow = 4
NumCols = 3

M2 = [[i + j for j in range(NumCols)] for i in range(NumRow)]
print(M2)

My_Array = np.array([1, 2, 3], dtype=np.float16)
print(My_Array)

# Matrix erstellen, shape 1dim für vector
my_zero_Array = np.zeros(shape=(10, 10), dtype=np.int32)
print(my_zero_Array)

# Matrix erstellen, shape 1dim für vector
my_reshape_Array = np.reshape(my_zero_Array, newshape=(2, 50))
print(my_reshape_Array)

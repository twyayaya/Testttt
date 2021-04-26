import numpy as np
import math
import csv
import numba as nb
with open('index_10\c1.weight10.csv', newline='') as csvfile:
    rows = csv.reader(csvfile, delimiter=',')
    fc1_weight = np.asarray(list(rows))  # input = 32*32*3

# print(fc1_weight.shape)

a = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [10, 9, 8, 7, 6],
    [5, 4, 3, 2, 1]

])
#print(a.size, type(a.size))
b = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
])


def laa(a):
    x = a.reshape((1, a.size))
    # print(x)
    return x

# 兩矩陣拉直後再加起來


def add(a, b):
    a = laa(a)
    b = laa(b)
    y = a.size + b.size
    x = np.zeros((1, y))
    for m in range(y):
        if 0 <= m < a.size:
            x[0][m] = (a[0][m])
        else:
            x[0][m] = b[0][m-a.size]
    return x


c = laa(a)

print("a = ", c)
print("b = ", b)
"""
ggg = b.size / c.size
print(type(ggg))
"""

# a = input矩陣 ,  b =輸出size大小(fc1=120,fc2=84,fc3=10) ,
# c = output[0][c] ,d = weight矩陣


@nb.jit()
def mulx(a, c, d):
    if a.size <= d.size and c <= a.size:
        s = int(d.size / a.size)
        b = a.size
        x = np.zeros((1, s))
        for k in range(c):
            h = k * b
            for n in range(b):
                x[0][k] += float(a[0][n]) * float(d[0][n+h])
        return x
    else:
        print("wrong matrix or c size")


y = mulx(b, 2, c)
print(y, "\n", "y shape = ", y.shape, "\n", "y size = ", y.size)
"""
h = add(a, b)
print(a.size)
print(b.size)
print("h = ", h, "h size = ", h.size)


c = laa(a)
print(c)

"""

"""
fc1w_0 = np.zeros((1, 400))

print(fc1_0)
print(fc1_0.size, fc1_0.shape)

for n in range(400):  # 0,1,2,....399
    fc1w_0[0][n] = fc1_weight[n][0]

# print(fc1w_0)
# print(fc1w_0.size, fc1w_0.shape)

x = np.zeros((1, 120))


@nb.jit()
def mm(a, b):
    for m in range(25):
        x[0][0] += a[0][m] * b[0][m]
    return x

gg = mm(c, fc1w_0)



for m in range(25):
    x[0][0] += c[0][m] * fc1w_0[0][m]


print(x)
print(x.size, x.shape)
"""

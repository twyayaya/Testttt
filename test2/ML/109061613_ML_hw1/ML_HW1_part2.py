import csv
import numpy as np
import math

a = 0
with open('training.csv', newline='') as csvfile, open('testing.csv', newline='') as csvfile2:
    rows = csv.reader(csvfile, delimiter=',')
    training_data = np.asarray(list(rows))
    rowss = csv.reader(csvfile2, delimiter=',')
    testing_data = np.asarray(list(rowss))

    # print(training_data)
    # print(testing_data[0][1])
    # print("~~~~~~~~~~~~~~~~~~~~~~~~")
# print(type(testing_data[0][1]))
col_number = training_data.shape[1]  # 14
row_number = training_data.shape[0]  # 124
print("col_number = ", col_number, " , ", "row_number = ", row_number)

col_number2 = testing_data.shape[1]  # 14
row_number2 = testing_data.shape[0]  # 54
print("col_number2 = ", col_number2, " , ", "row_number2 = ", row_number2)
# 轉成 float
a = training_data.astype("float")
b = testing_data.astype("float")
##
print("---------")


def meann(a, b, c):  # a:training.csv , b: type , c:特徵
    t1 = 0
    t2 = 0
    t3 = 0
    t11 = np.zeros((1, 41))
    t22 = np.zeros((1, 53))
    t33 = np.zeros((1, 30))
    for n in range(row_number):
        hh = a[n][0]

        if(hh == 1 and b == 1):
            # print(a[n])
            t11[0][n] = a[n][c]
            t1 += 1

        if(hh == 2 and b == 2):
            # print(a[n])
            t22[0][n-41] = a[n][c]
            t2 += 1

        if(hh == 3 and b == 3):
            # print(a[n])
            t33[0][n-94] = a[n][c]
            # print(t33)
            t3 += 1
    #print(t1, t2, t3)
    #t3_mean = np.mean(t33)
    # print(t3_mean)
    if(b == 1):
        m = np.mean(t11)
        return m
    if(b == 2):
        mm = np.mean(t22)
        return mm
    if(b == 3):
        mmm = np.mean(t33)
        return mmm


def varr(a, b, c):  # a:training.csv , b: type , c:特徵
    t1 = 0
    t2 = 0
    t3 = 0
    t11 = np.zeros((1, 41))
    t22 = np.zeros((1, 53))
    t33 = np.zeros((1, 30))
    for n in range(row_number):
        hh = a[n][0]

        if(hh == 1 and b == 1):
            # print(a[n])
            t11[0][n] = a[n][c]
            t1 += 1

        if(hh == 2 and b == 2):
            # print(a[n])
            t22[0][n-41] = a[n][c]
            t2 += 1

        if(hh == 3 and b == 3):
            # print(a[n])
            t33[0][n-94] = a[n][c]
            # print(t33)
            t3 += 1
    #print(t1, t2, t3)
    #t3_mean = np.mean(t33)
    # print(t3_mean)
    if(b == 1):
        m = np.var(t11)
        return m
    if(b == 2):
        mm = np.var(t22)
        return mm
    if(b == 3):
        mmm = np.var(t33)
        return mmm


"""
print("----------")
test = meann(a, 1, 1)
print(test)
print("----------")
v = varr(a, 1, 1)
print(v)
"""


def n(x, m, v):  # x:input m:mean v:var
    norm = (1/((2*math.pi*v)**(1/2)))*math.exp((-1/(2*v))*((x-m)**2))

    return norm


# type1 特徵1~13 mean
t1c1m = meann(a, 1, 1)
t1c2m = meann(a, 1, 2)
t1c3m = meann(a, 1, 3)
t1c4m = meann(a, 1, 4)
t1c5m = meann(a, 1, 5)
t1c6m = meann(a, 1, 6)
t1c7m = meann(a, 1, 7)
t1c8m = meann(a, 1, 8)
t1c9m = meann(a, 1, 9)
t1c10m = meann(a, 1, 10)
t1c11m = meann(a, 1, 11)
t1c12m = meann(a, 1, 12)
t1c13m = meann(a, 1, 13)
# type2 特徵1~13 mean
t2c1m = meann(a, 2, 1)
t2c2m = meann(a, 2, 2)
t2c3m = meann(a, 2, 3)
t2c4m = meann(a, 2, 4)
t2c5m = meann(a, 2, 5)
t2c6m = meann(a, 2, 6)
t2c7m = meann(a, 2, 7)
t2c8m = meann(a, 2, 8)
t2c9m = meann(a, 2, 9)
t2c10m = meann(a, 2, 10)
t2c11m = meann(a, 2, 11)
t2c12m = meann(a, 2, 12)
t2c13m = meann(a, 2, 13)
# type3 特徵1~13 mean
t3c1m = meann(a, 3, 1)
t3c2m = meann(a, 3, 2)
t3c3m = meann(a, 3, 3)
t3c4m = meann(a, 3, 4)
t3c5m = meann(a, 3, 5)
t3c6m = meann(a, 3, 6)
t3c7m = meann(a, 3, 7)
t3c8m = meann(a, 3, 8)
t3c9m = meann(a, 3, 9)
t3c10m = meann(a, 3, 10)
t3c11m = meann(a, 3, 11)
t3c12m = meann(a, 3, 12)
t3c13m = meann(a, 3, 13)
##############################################
# type1 特徵1~13 var
t1c1v = varr(a, 1, 1)
t1c2v = varr(a, 1, 2)
t1c3v = varr(a, 1, 3)
t1c4v = varr(a, 1, 4)
t1c5v = varr(a, 1, 5)
t1c6v = varr(a, 1, 6)
t1c7v = varr(a, 1, 7)
t1c8v = varr(a, 1, 8)
t1c9v = varr(a, 1, 9)
t1c10v = varr(a, 1, 10)
t1c11v = varr(a, 1, 11)
t1c12v = varr(a, 1, 12)
t1c13v = varr(a, 1, 13)
# type2 特徵1~13 var
t2c1v = varr(a, 2, 1)
t2c2v = varr(a, 2, 2)
t2c3v = varr(a, 2, 3)
t2c4v = varr(a, 2, 4)
t2c5v = varr(a, 2, 5)
t2c6v = varr(a, 2, 6)
t2c7v = varr(a, 2, 7)
t2c8v = varr(a, 2, 8)
t2c9v = varr(a, 2, 9)
t2c10v = varr(a, 2, 10)
t2c11v = varr(a, 2, 11)
t2c12v = varr(a, 2, 12)
t2c13v = varr(a, 2, 13)
# type3 特徵1~13 var
t3c1v = varr(a, 3, 1)
t3c2v = varr(a, 3, 2)
t3c3v = varr(a, 3, 3)
t3c4v = varr(a, 3, 4)
t3c5v = varr(a, 3, 5)
t3c6v = varr(a, 3, 6)
t3c7v = varr(a, 3, 7)
t3c8v = varr(a, 3, 8)
t3c9v = varr(a, 3, 9)
t3c10v = varr(a, 3, 10)
t3c11v = varr(a, 3, 11)
t3c12v = varr(a, 3, 12)
t3c13v = varr(a, 3, 13)
###########################################
##
# posterior probability = likelihood *prior probability
##########################################
count = 0
s = np.zeros((1, 3))
for m in range(54):
    type1p = n(a[m][1], t1c1m, t1c1v)*n(a[m][2], t1c2m, t1c2v)*n(a[m][3], t1c3m, t1c3v)*n(a[m][4], t1c4m, t1c4v)*n(a[m][5], t1c5m, t1c5v)*n(a[m][6], t1c6m, t1c6v) * \
        n(a[m][7], t1c7m, t1c7v)*n(a[m][8], t1c8m, t1c8v)*n(a[m][9], t1c9m, t1c9v) * \
        n(a[m][10], t1c10m, t1c10v)*n(a[m][11], t1c11m, t1c11v) * \
        n(a[m][12], t1c12m, t1c12v)*n(a[m][13], t1c13m, t1c13v)*(41/124)

    type2p = n(a[m][1], t2c1m, t2c1v)*n(a[m][2], t2c2m, t2c2v)*n(a[m][3], t2c3m, t2c3v)*n(a[m][4], t2c4m, t2c4v)*n(a[m][5], t2c5m, t2c5v)*n(a[m][6], t2c6m, t2c6v) * \
        n(a[m][7], t2c7m, t2c7v)*n(a[m][8], t2c8m, t2c8v)*n(a[m][9], t2c9m, t2c9v) * \
        n(a[m][10], t2c10m, t2c10v)*n(a[m][11], t2c11m, t2c11v) * \
        n(a[m][12], t2c12m, t2c12v)*n(a[m][13], t2c13m, t1c13v)*(53/124)

    type3p = n(a[m][1], t3c1m, t3c1v)*n(a[m][2], t3c2m, t3c2v)*n(a[m][3], t3c3m, t3c3v)*n(a[m][4], t3c4m, t3c4v)*n(a[m][5], t3c5m, t3c5v)*n(a[m][6], t3c6m, t3c6v) * \
        n(a[m][7], t3c7m, t3c7v)*n(a[m][8], t3c8m, t3c8v)*n(a[m][9], t3c9m, t3c9v) * \
        n(a[m][10], t3c10m, t3c10v)*n(a[m][11], t3c11m, t3c11v) * \
        n(a[m][12], t3c12m, t3c12v)*n(a[m][13], t3c13m, t3c13v)*(30/124)

    s[0][0] = type1p
    s[0][1] = type2p
    s[0][2] = type3p

    w = np.argmax(s) + 1  # 取max

    if (0 <= m < 18 and a[m][0] == w):
        count += 1
    if (18 <= m < 36 and a[m][0] == w):
        count += 1
    if (36 <= m < 54 and a[m][0] == w):
        count += 1

acc = (count/54)*100

print("!!!!!!!!!!!!!!!!!!!!!!")
print(acc, "%")

print("!!!!!!!!!!!!!!!!!!!!!!")

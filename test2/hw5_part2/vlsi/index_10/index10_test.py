import csv
import random
import numpy as np
import math

from test2 import la
from test2 import laa
from test2 import convtest
from test2 import test

test()
with open('index_10\input10.csv', newline='') as csvfile, open('index_10\conv1.weight10.csv', newline='') as csvfile2:
    rows = csv.reader(csvfile, delimiter=',')
    inputdata = np.asarray(list(rows))  # input = 32*32*3
    # print("len of (input.csv) = ", len(inputdata), "type = ", type(inputdata))
    rowss = csv.reader(csvfile2, delimiter=',')
    conv1weight = np.asarray(list(rowss))


# new array for input R G B
number = int(inputdata.shape[0])
single = int(inputdata.shape[0]/3)  # 1024 = 32^2
l = int(math.sqrt(single))  # input 單邊長 : 32
i_r = np.zeros((1, single))
i_g = np.zeros((1, single))
i_b = np.zeros((1, single))

# print("i_r = ",i_r.size)
# print("single = ",single)
# print("number = ",number)
##

# 直行換成橫列的, 分類input to RBG

for r in range(number):
    if 0 <= r < single:
        i_r[0][r] = inputdata[r][0]

    elif single <= r < (2*single):
        i_g[0][r-single] = inputdata[r][0]

    elif 2*single <= r < inputdata.shape[0]:
        i_b[0][r-(2*single)] = inputdata[r][0]


# conv1weight
conv1weight_number = int(conv1weight.shape[0])  # 5*5*3*6 = 450
conv1cut = int(conv1weight_number/18)  # 5*5 = 25
#
conv1w_r0 = np.zeros((1, conv1cut))
conv1w_g0 = np.zeros((1, conv1cut))
conv1w_b0 = np.zeros((1, conv1cut))
#
conv1w_r1 = np.zeros((1, conv1cut))
conv1w_g1 = np.zeros((1, conv1cut))
conv1w_b1 = np.zeros((1, conv1cut))
#
conv1w_r2 = np.zeros((1, conv1cut))
conv1w_g2 = np.zeros((1, conv1cut))
conv1w_b2 = np.zeros((1, conv1cut))
#
conv1w_r3 = np.zeros((1, conv1cut))
conv1w_g3 = np.zeros((1, conv1cut))
conv1w_b3 = np.zeros((1, conv1cut))
#
conv1w_r4 = np.zeros((1, conv1cut))
conv1w_g4 = np.zeros((1, conv1cut))
conv1w_b4 = np.zeros((1, conv1cut))
#
conv1w_r5 = np.zeros((1, conv1cut))
conv1w_g5 = np.zeros((1, conv1cut))
conv1w_b5 = np.zeros((1, conv1cut))


# print(conv1w_r1)


for n in range(conv1weight_number):
    if 0 <= n < conv1cut:
        conv1w_r0[0][n] = conv1weight[n][0]

    elif 1*conv1cut <= n < 2*conv1cut:
        conv1w_g0[0][n-conv1cut] = conv1weight[n][0]

    elif 2*conv1cut <= n < 3*conv1cut:
        conv1w_b0[0][n-2*conv1cut] = conv1weight[n][0]

    elif 3*conv1cut <= n < 4*conv1cut:
        conv1w_r1[0][n-3*conv1cut] = conv1weight[n][0]

    elif 4*conv1cut <= n < 5*conv1cut:
        conv1w_g1[0][n-4*conv1cut] = conv1weight[n][0]

    elif 5*conv1cut <= n < 6*conv1cut:
        conv1w_b1[0][n-5*conv1cut] = conv1weight[n][0]

    elif 6*conv1cut <= n < 7*conv1cut:
        conv1w_r2[0][n-6*conv1cut] = conv1weight[n][0]

    elif 7*conv1cut <= n < 8*conv1cut:
        conv1w_g2[0][n-7*conv1cut] = conv1weight[n][0]

    elif 8*conv1cut <= n < 9*conv1cut:
        conv1w_b2[0][n-8*conv1cut] = conv1weight[n][0]

    elif 9*conv1cut <= n < 10*conv1cut:
        conv1w_r3[0][n-9*conv1cut] = conv1weight[n][0]

    elif 10*conv1cut <= n < 11*conv1cut:
        conv1w_g3[0][n-10*conv1cut] = conv1weight[n][0]

    elif 11*conv1cut <= n < 12*conv1cut:
        conv1w_b3[0][n-11*conv1cut] = conv1weight[n][0]

    elif 12*conv1cut <= n < 13*conv1cut:
        conv1w_r4[0][n-12*conv1cut] = conv1weight[n][0]

    elif 13*conv1cut <= n < 14*conv1cut:
        conv1w_g4[0][n-13*conv1cut] = conv1weight[n][0]

    elif 14*conv1cut <= n < 15*conv1cut:
        conv1w_b4[0][n-14*conv1cut] = conv1weight[n][0]

    elif 15*conv1cut <= n < 16*conv1cut:
        conv1w_r5[0][n-15*conv1cut] = conv1weight[n][0]

    elif 16*conv1cut <= n < 17*conv1cut:
        conv1w_g5[0][n-16*conv1cut] = conv1weight[n][0]

    elif 17*conv1cut <= n < 18*conv1cut:
        conv1w_b5[0][n-17*conv1cut] = conv1weight[n][0]


print(conv1w_b5)


# output activation

out0 = convtest(i_r, conv1w_r0) + convtest(i_g, conv1w_g0) + \
    convtest(i_b, conv1w_b0)
out1 = convtest(i_r, conv1w_r1) + convtest(i_g, conv1w_g1) + \
    convtest(i_b, conv1w_b1)
out2 = convtest(i_r, conv1w_r2) + convtest(i_g, conv1w_g2) + \
    convtest(i_b, conv1w_b2)
out3 = convtest(i_r, conv1w_r3) + convtest(i_g, conv1w_g3) + \
    convtest(i_b, conv1w_b3)
out4 = convtest(i_r, conv1w_r4) + convtest(i_g, conv1w_g4) + \
    convtest(i_b, conv1w_b4)
out5 = convtest(i_r, conv1w_r5) + convtest(i_g, conv1w_g5) + \
    convtest(i_b, conv1w_b5)

print("-------------OUT---------")
print(i_r)
print(i_r.size)
print(conv1w_r0)
print(convtest(i_r, conv1w_r0))
print(out0)

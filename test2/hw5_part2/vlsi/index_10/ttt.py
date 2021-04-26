import numpy as np
import math
import csv
import numba as nb

################scale##############
input_scale = 0.00784313725490196
conv1_output_scale = 0.0005651640144149312
conv2_output_scale = 0.003286375832980571
fc1_output_scale = 0.002551974189109988
fc2_output_scale = 0.0028480268876474738
fc3_output_scale = 0.002825382723164684
###################################

############################

# clip  min = -128, max =127


def clip(a):
    a[a > 127] = 127
    a[a < -128] = -128
    return a

# 先round 再限制-128~127


def qq(a):
    b = a.round()
    b = clip(b)
    return b

# a = input矩陣 ,  b =輸出size大小(fc1=120,fc2=84,fc3=10) ,
# c = output[0][c] ,d = weight矩陣


def mulx(a, c, d):
    if a.size <= d.size and c <= a.size:
        s = int(d.size / a.size)
        b = a.size
        x = np.zeros((1, s))
        for k in range(c):
            h = k * b
            for n in range(b):
                x[0][k] += a[0][n] * float(d[0][n+h])
        return x
    else:
        print("wrong matrix or c size")


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

# mn to nn矩陣


def mntonn(a):
    # print(a.shape[1])
    # print(math.sqrt(a.shape[1]), type(math.sqrt(a.shape[1])))
    if math.sqrt(a.shape[1]).is_integer() == 1:
        s = int(math.sqrt(a.shape[1]))
        x = np.zeros((s, s))
        k = 0
        for m in range(s):
            for n in range(s):
                x[m][n] = a[0][k]
                k += 1
        return x
    else:
        print("wrong number")


def la(a, b, c, d):  # a:原本矩陣 b:weight c:a的c列 d:a的d行 ->把a拉成b矩陣內元素數的一列
    x = np.zeros((b.shape[0], b.shape[1]))
    for n in range(b.shape[0]):
        for k in range(b.shape[1]):
            x[n][k] = a[n+c][k+d]

    return x.reshape((1, b.size))


def laa(a):
    x = a.reshape((1, a.size))
    # print(x)
    return x


def convtest(a, b):  # a:input b:weight
    if a.size >= b.size:
        j_size = (a.shape[0] - b.shape[0] + 1)  # output長or寬 = 3
        j = np.zeros((j_size, j_size))  # output矩陣 = 3*3
        z = laa(b)  # 拉直b(weight)

        for m in range(j_size):  # 列
            # print("m = ", m)
            for n in range(j_size):  # 行
                # print("n = ", n)
                y = la(a, b, m, n)
            # print("y = ", y)  # 拉直a要被conv的部分
                for u in range(z.size):  # j矩陣內元素個數(j_size*j_size)
                    j[m][n] += y[0][u] * z[0][u]

        return j
    else:
        print("wrong number:a must >= b")


def lacd(a, b, c, d):  # a:原本矩陣 b:weight c:a的c列 d:a的d行 ->把a拉成b矩陣內元素數的一列
    x = np.zeros((b.shape[0], b.shape[1]))
    for n in range(b.shape[0]):
        for k in range(b.shape[1]):
            x[n][k] = a[n+2*c][k+2*d]

    return x.reshape((1, b.size))


def pool(a, b):
    out_size = int(a.shape[0]/2)  # 3 shape[0]:列
    if (a.shape[0] % 2) == 0.0:
        s = np.zeros((out_size, out_size))  # 3*3
        z = laa(b)
        count = out_size  # 0 1 2
        for m in range(count):  # 列
            for n in range(count):  # 行
                y = lacd(a, b, m, n)  # 拉直a要被conv的部分
                s[m][n] = y.max()
        return s
    else:
        print("wrong input")


Maxpooling_matrix = np.array([
    [1, 1],
    [1, 1]
])
#######################################################


with open('input10.csv', newline='') as csvfile, \
        open('conv1.weight10.csv', newline='') as csvfile2, \
    open('conv2.weight10.csv', newline='') as csvfile3, \
        open('c1.weight10.csv', newline='') as csvfile4, \
    open('c2.weight10.csv', newline='') as csvfile5,\
        open('c3.weight10.csv', newline='') as csvfile6, \
    open('c3.bias10.csv', newline='') as csvfile7, \
        open('output10.csv', newline='') as csvfile8:
    rows = csv.reader(csvfile, delimiter=',')
    inputdata = np.array(list(rows))  # input = 32*32*3
    #
    rowss = csv.reader(csvfile2, delimiter=',')
    conv1weight = np.array(list(rowss))  # 5*5*3*6
    #
    rowsss = csv.reader(csvfile3, delimiter=',')
    conv2weight = np.array(list(rowsss))  # 5*5*6*16
    #
    rowssss = csv.reader(csvfile4, delimiter=',')
    fc1_weight = np.array(list(rowssss))  # 5*5*6*16
    #
    rowsssss = csv.reader(csvfile5, delimiter=',')
    fc2_weight = np.array(list(rowsssss))  # 5*5*6*16
    #
    rowssssss = csv.reader(csvfile6, delimiter=',')
    fc3_weight = np.array(list(rowssssss))  # 5*5*6*16
    #
    rowsssssss = csv.reader(csvfile7, delimiter=',')
    fc3_bias = np.array(list(rowsssssss))  # 5*5*6*16
    #
    rowssssssss = csv.reader(csvfile8, delimiter=',')
    output = np.array(list(rowssssssss))  # 5*5*6*16

# new array for input R G B
number = int(inputdata.shape[0])
single = int(inputdata.shape[0]/3)  # 1024 = 32^2
l = int(math.sqrt(single))  # input 單邊長 : 32
i_r = np.zeros((1, single))
i_g = np.zeros((1, single))
i_b = np.zeros((1, single))

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

# 直行換成橫列的, 分類input to RBG

for r in range(number):
    if 0 <= r < single:
        i_r[0][r] = inputdata[r][0]

    elif single <= r < (2*single):
        i_g[0][r-single] = inputdata[r][0]

    elif 2*single <= r < inputdata.shape[0]:
        i_b[0][r-(2*single)] = inputdata[r][0]

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


# [0][m] to [n][n]
i_r = mntonn(i_r)
i_g = mntonn(i_g)
i_b = mntonn(i_b)
conv1w_r0 = mntonn(conv1w_r0)
conv1w_r1 = mntonn(conv1w_r1)
conv1w_r2 = mntonn(conv1w_r2)
conv1w_r3 = mntonn(conv1w_r3)
conv1w_r4 = mntonn(conv1w_r4)
conv1w_r5 = mntonn(conv1w_r5)
conv1w_g0 = mntonn(conv1w_g0)
conv1w_g1 = mntonn(conv1w_g1)
conv1w_g2 = mntonn(conv1w_g2)
conv1w_g3 = mntonn(conv1w_g3)
conv1w_g4 = mntonn(conv1w_g4)
conv1w_g5 = mntonn(conv1w_g5)
conv1w_b0 = mntonn(conv1w_b0)
conv1w_b1 = mntonn(conv1w_b1)
conv1w_b2 = mntonn(conv1w_b2)
conv1w_b3 = mntonn(conv1w_b3)
conv1w_b4 = mntonn(conv1w_b4)
conv1w_b5 = mntonn(conv1w_b5)

###input 量化 ###
i_r = qq(i_r / input_scale)
i_g = qq(i_g / input_scale)
i_b = qq(i_b / input_scale)

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

print("conv1 output shape : ", out2.shape)

##conv1 Relu##
out0 = np.maximum(out0, 0)
out1 = np.maximum(out1, 0)
out2 = np.maximum(out2, 0)
out3 = np.maximum(out3, 0)
out4 = np.maximum(out4, 0)
out5 = np.maximum(out5, 0)


# conv1 MaxPooling =  input of conv2
out0 = pool(out0, Maxpooling_matrix)
out1 = pool(out1, Maxpooling_matrix)
out2 = pool(out2, Maxpooling_matrix)
out3 = pool(out3, Maxpooling_matrix)
out4 = pool(out4, Maxpooling_matrix)
out5 = pool(out5, Maxpooling_matrix)

### conv1 out 量化 ###
out0 = qq(out0 * 6*conv1_output_scale)
out1 = qq(out1 * 6*conv1_output_scale)
out2 = qq(out2 * 6*conv1_output_scale)
out3 = qq(out3 * 6*conv1_output_scale)
out4 = qq(out4 * 6 * conv1_output_scale)
out5 = qq(out5 * 6 * conv1_output_scale)


print("after first Maxpooling shape : ", out2.shape)
# conv2的output

conv2w0_out = np.zeros((10, 10))
conv2w1_out = np.zeros((10, 10))
conv2w2_out = np.zeros((10, 10))
conv2w3_out = np.zeros((10, 10))
conv2w4_out = np.zeros((10, 10))
conv2w5_out = np.zeros((10, 10))
conv2w6_out = np.zeros((10, 10))
conv2w7_out = np.zeros((10, 10))
conv2w8_out = np.zeros((10, 10))
conv2w9_out = np.zeros((10, 10))
conv2w10_out = np.zeros((10, 10))
conv2w11_out = np.zeros((10, 10))
conv2w12_out = np.zeros((10, 10))
conv2w13_out = np.zeros((10, 10))
conv2w14_out = np.zeros((10, 10))
conv2w15_out = np.zeros((10, 10))
##

# conv2 weight 5*5 95個
conv2w0 = np.zeros((1, 25))
conv2w1 = np.zeros((1, 25))
conv2w2 = np.zeros((1, 25))
conv2w3 = np.zeros((1, 25))
conv2w4 = np.zeros((1, 25))
conv2w5 = np.zeros((1, 25))
conv2w6 = np.zeros((1, 25))
conv2w7 = np.zeros((1, 25))
conv2w8 = np.zeros((1, 25))
conv2w9 = np.zeros((1, 25))
conv2w10 = np.zeros((1, 25))
conv2w11 = np.zeros((1, 25))
conv2w12 = np.zeros((1, 25))
conv2w13 = np.zeros((1, 25))
conv2w14 = np.zeros((1, 25))
conv2w15 = np.zeros((1, 25))
conv2w16 = np.zeros((1, 25))
conv2w17 = np.zeros((1, 25))
conv2w18 = np.zeros((1, 25))
conv2w19 = np.zeros((1, 25))
conv2w20 = np.zeros((1, 25))
conv2w21 = np.zeros((1, 25))
conv2w22 = np.zeros((1, 25))
conv2w23 = np.zeros((1, 25))
conv2w24 = np.zeros((1, 25))
conv2w25 = np.zeros((1, 25))
conv2w26 = np.zeros((1, 25))
conv2w27 = np.zeros((1, 25))
conv2w28 = np.zeros((1, 25))
conv2w29 = np.zeros((1, 25))
conv2w30 = np.zeros((1, 25))
conv2w31 = np.zeros((1, 25))
conv2w32 = np.zeros((1, 25))
conv2w33 = np.zeros((1, 25))
conv2w34 = np.zeros((1, 25))
conv2w35 = np.zeros((1, 25))
conv2w36 = np.zeros((1, 25))
conv2w37 = np.zeros((1, 25))
conv2w38 = np.zeros((1, 25))
conv2w39 = np.zeros((1, 25))
conv2w40 = np.zeros((1, 25))
conv2w41 = np.zeros((1, 25))
conv2w42 = np.zeros((1, 25))
conv2w43 = np.zeros((1, 25))
conv2w44 = np.zeros((1, 25))
conv2w45 = np.zeros((1, 25))
conv2w46 = np.zeros((1, 25))
conv2w47 = np.zeros((1, 25))
conv2w48 = np.zeros((1, 25))
conv2w49 = np.zeros((1, 25))
conv2w50 = np.zeros((1, 25))
conv2w51 = np.zeros((1, 25))
conv2w52 = np.zeros((1, 25))
conv2w53 = np.zeros((1, 25))
conv2w54 = np.zeros((1, 25))
conv2w55 = np.zeros((1, 25))
conv2w56 = np.zeros((1, 25))
conv2w57 = np.zeros((1, 25))
conv2w58 = np.zeros((1, 25))
conv2w59 = np.zeros((1, 25))
conv2w60 = np.zeros((1, 25))
conv2w61 = np.zeros((1, 25))
conv2w62 = np.zeros((1, 25))
conv2w63 = np.zeros((1, 25))
conv2w64 = np.zeros((1, 25))
conv2w65 = np.zeros((1, 25))
conv2w66 = np.zeros((1, 25))
conv2w67 = np.zeros((1, 25))
conv2w68 = np.zeros((1, 25))
conv2w69 = np.zeros((1, 25))
conv2w70 = np.zeros((1, 25))
conv2w71 = np.zeros((1, 25))
conv2w72 = np.zeros((1, 25))
conv2w73 = np.zeros((1, 25))
conv2w74 = np.zeros((1, 25))
conv2w75 = np.zeros((1, 25))
conv2w76 = np.zeros((1, 25))
conv2w77 = np.zeros((1, 25))
conv2w78 = np.zeros((1, 25))
conv2w79 = np.zeros((1, 25))
conv2w80 = np.zeros((1, 25))
conv2w81 = np.zeros((1, 25))
conv2w82 = np.zeros((1, 25))
conv2w83 = np.zeros((1, 25))
conv2w84 = np.zeros((1, 25))
conv2w85 = np.zeros((1, 25))
conv2w86 = np.zeros((1, 25))
conv2w87 = np.zeros((1, 25))
conv2w88 = np.zeros((1, 25))
conv2w89 = np.zeros((1, 25))
conv2w90 = np.zeros((1, 25))
conv2w91 = np.zeros((1, 25))
conv2w92 = np.zeros((1, 25))
conv2w93 = np.zeros((1, 25))
conv2w94 = np.zeros((1, 25))
conv2w95 = np.zeros((1, 25))
##

# 分配conv2的2400個數字給 6*16個矩陣 一個矩陣5*5

n25 = 25  # conv2 weight單個矩陣大小 = 5*5
for n in range(conv2weight.size):   # conv2weight.size = 2400 -> 0,1,2...,2399
    if 0 <= n < n25:
        conv2w0[0][n] = conv2weight[n][0]
    elif 1*n25 <= n < 2*n25:
        conv2w1[0][n - 1*n25] = conv2weight[n][0]
    elif 2*n25 <= n < 3*n25:
        conv2w2[0][n - 2*n25] = conv2weight[n][0]
    elif 3*n25 <= n < 4*n25:
        conv2w3[0][n - 3*n25] = conv2weight[n][0]
    elif 4*n25 <= n < 5*n25:
        conv2w4[0][n - 4*n25] = conv2weight[n][0]
    elif 5*n25 <= n < 6*n25:
        conv2w5[0][n - 5*n25] = conv2weight[n][0]
    elif 6*n25 <= n < 7*n25:
        conv2w6[0][n - 6*n25] = conv2weight[n][0]
    elif 7*n25 <= n < 8*n25:
        conv2w7[0][n - 7*n25] = conv2weight[n][0]
    elif 8*n25 <= n < 9*n25:
        conv2w8[0][n - 8*n25] = conv2weight[n][0]
    elif 9*n25 <= n < 10*n25:
        conv2w9[0][n - 9*n25] = conv2weight[n][0]
    elif 10*n25 <= n < 11*n25:
        conv2w10[0][n - 10*n25] = conv2weight[n][0]
    elif 11*n25 <= n < 12*n25:
        conv2w11[0][n - 11*n25] = conv2weight[n][0]
    elif 12*n25 <= n < 13*n25:
        conv2w12[0][n - 12*n25] = conv2weight[n][0]
    elif 13*n25 <= n < 14*n25:
        conv2w13[0][n - 13*n25] = conv2weight[n][0]
    elif 14*n25 <= n < 15*n25:
        conv2w14[0][n - 14*n25] = conv2weight[n][0]
    elif 15*n25 <= n < 16*n25:
        conv2w15[0][n - 15*n25] = conv2weight[n][0]
    elif 16*n25 <= n < 17*n25:
        conv2w16[0][n - 16*n25] = conv2weight[n][0]
    elif 17*n25 <= n < 18*n25:
        conv2w17[0][n - 17*n25] = conv2weight[n][0]
    elif 18*n25 <= n < 19*n25:
        conv2w18[0][n - 18*n25] = conv2weight[n][0]
    elif 19*n25 <= n < 20*n25:
        conv2w19[0][n - 19*n25] = conv2weight[n][0]
    elif 20*n25 <= n < 21*n25:
        conv2w20[0][n - 20*n25] = conv2weight[n][0]
    elif 21*n25 <= n < 22*n25:
        conv2w21[0][n - 21*n25] = conv2weight[n][0]
    elif 22*n25 <= n < 23*n25:
        conv2w22[0][n - 22*n25] = conv2weight[n][0]
    elif 23*n25 <= n < 24*n25:
        conv2w23[0][n - 23*n25] = conv2weight[n][0]
    elif 24*n25 <= n < 25*n25:
        conv2w24[0][n - 24*n25] = conv2weight[n][0]
    elif 25*n25 <= n < 26*n25:
        conv2w25[0][n - 25*n25] = conv2weight[n][0]
    elif 26*n25 <= n < 27*n25:
        conv2w26[0][n - 26*n25] = conv2weight[n][0]
    elif 27*n25 <= n < 28*n25:
        conv2w27[0][n - 27*n25] = conv2weight[n][0]
    elif 28*n25 <= n < 29*n25:
        conv2w28[0][n - 28*n25] = conv2weight[n][0]
    elif 29*n25 <= n < 30*n25:
        conv2w29[0][n - 29*n25] = conv2weight[n][0]
    elif 30*n25 <= n < 31*n25:
        conv2w30[0][n - 30*n25] = conv2weight[n][0]
    elif 31*n25 <= n < 32*n25:
        conv2w31[0][n - 31*n25] = conv2weight[n][0]
    elif 32*n25 <= n < 33*n25:
        conv2w32[0][n - 32*n25] = conv2weight[n][0]
    elif 33*n25 <= n < 34*n25:
        conv2w33[0][n - 33*n25] = conv2weight[n][0]
    elif 34*n25 <= n < 35*n25:
        conv2w34[0][n - 34*n25] = conv2weight[n][0]
    elif 35*n25 <= n < 36*n25:
        conv2w35[0][n - 35*n25] = conv2weight[n][0]
    elif 36*n25 <= n < 37*n25:
        conv2w36[0][n - 36*n25] = conv2weight[n][0]
    elif 37*n25 <= n < 38*n25:
        conv2w37[0][n - 37*n25] = conv2weight[n][0]
    elif 38*n25 <= n < 39*n25:
        conv2w38[0][n - 38*n25] = conv2weight[n][0]
    elif 39*n25 <= n < 40*n25:
        conv2w39[0][n - 39*n25] = conv2weight[n][0]
    elif 40*n25 <= n < 41*n25:
        conv2w40[0][n - 40*n25] = conv2weight[n][0]
    elif 41*n25 <= n < 42*n25:
        conv2w41[0][n - 41*n25] = conv2weight[n][0]
    elif 42*n25 <= n < 43*n25:
        conv2w42[0][n - 42*n25] = conv2weight[n][0]
    elif 43*n25 <= n < 44*n25:
        conv2w43[0][n - 43*n25] = conv2weight[n][0]
    elif 44*n25 <= n < 45*n25:
        conv2w44[0][n - 44*n25] = conv2weight[n][0]
    elif 45*n25 <= n < 46*n25:
        conv2w45[0][n - 45*n25] = conv2weight[n][0]
    elif 46*n25 <= n < 47*n25:
        conv2w46[0][n - 46*n25] = conv2weight[n][0]
    elif 47*n25 <= n < 48*n25:
        conv2w47[0][n - 47*n25] = conv2weight[n][0]
    elif 48*n25 <= n < 49*n25:
        conv2w48[0][n - 48*n25] = conv2weight[n][0]
    elif 49*n25 <= n < 50*n25:
        conv2w49[0][n - 49*n25] = conv2weight[n][0]
    elif 50*n25 <= n < 51*n25:
        conv2w50[0][n - 50*n25] = conv2weight[n][0]
    elif 51*n25 <= n < 52*n25:
        conv2w51[0][n - 51*n25] = conv2weight[n][0]
    elif 52*n25 <= n < 53*n25:
        conv2w52[0][n - 52*n25] = conv2weight[n][0]
    elif 53*n25 <= n < 54*n25:
        conv2w53[0][n - 53*n25] = conv2weight[n][0]
    elif 54*n25 <= n < 55*n25:
        conv2w54[0][n - 54*n25] = conv2weight[n][0]
    elif 55*n25 <= n < 56*n25:
        conv2w55[0][n - 55*n25] = conv2weight[n][0]
    elif 56*n25 <= n < 57*n25:
        conv2w56[0][n - 56*n25] = conv2weight[n][0]
    elif 57*n25 <= n < 58*n25:
        conv2w57[0][n - 57*n25] = conv2weight[n][0]
    elif 58*n25 <= n < 59*n25:
        conv2w58[0][n - 58*n25] = conv2weight[n][0]
    elif 59*n25 <= n < 60*n25:
        conv2w59[0][n - 59*n25] = conv2weight[n][0]
    elif 60*n25 <= n < 61*n25:
        conv2w60[0][n - 60*n25] = conv2weight[n][0]
    elif 61*n25 <= n < 62*n25:
        conv2w61[0][n - 61*n25] = conv2weight[n][0]
    elif 62*n25 <= n < 63*n25:
        conv2w62[0][n - 62*n25] = conv2weight[n][0]
    elif 63*n25 <= n < 64*n25:
        conv2w63[0][n - 63*n25] = conv2weight[n][0]
    elif 64*n25 <= n < 65*n25:
        conv2w64[0][n - 64*n25] = conv2weight[n][0]
    elif 65*n25 <= n < 66*n25:
        conv2w65[0][n - 65*n25] = conv2weight[n][0]
    elif 66*n25 <= n < 67*n25:
        conv2w66[0][n - 66*n25] = conv2weight[n][0]
    elif 67*n25 <= n < 68*n25:
        conv2w67[0][n - 67*n25] = conv2weight[n][0]
    elif 68*n25 <= n < 69*n25:
        conv2w68[0][n - 68*n25] = conv2weight[n][0]
    elif 69*n25 <= n < 70*n25:
        conv2w69[0][n - 69*n25] = conv2weight[n][0]
    elif 70*n25 <= n < 71*n25:
        conv2w70[0][n - 70*n25] = conv2weight[n][0]
    elif 71*n25 <= n < 72*n25:
        conv2w71[0][n - 71*n25] = conv2weight[n][0]
    elif 72*n25 <= n < 73*n25:
        conv2w72[0][n - 72*n25] = conv2weight[n][0]
    elif 73*n25 <= n < 74*n25:
        conv2w73[0][n - 73*n25] = conv2weight[n][0]
    elif 74*n25 <= n < 75*n25:
        conv2w74[0][n - 74*n25] = conv2weight[n][0]
    elif 75*n25 <= n < 76*n25:
        conv2w75[0][n - 75*n25] = conv2weight[n][0]
    elif 76*n25 <= n < 77*n25:
        conv2w76[0][n - 76*n25] = conv2weight[n][0]
    elif 77*n25 <= n < 78*n25:
        conv2w77[0][n - 77*n25] = conv2weight[n][0]
    elif 78*n25 <= n < 79*n25:
        conv2w78[0][n - 78*n25] = conv2weight[n][0]
    elif 79*n25 <= n < 80*n25:
        conv2w79[0][n - 79*n25] = conv2weight[n][0]
    elif 80*n25 <= n < 81*n25:
        conv2w80[0][n - 80*n25] = conv2weight[n][0]
    elif 81*n25 <= n < 82*n25:
        conv2w81[0][n - 81*n25] = conv2weight[n][0]
    elif 82*n25 <= n < 83*n25:
        conv2w82[0][n - 82*n25] = conv2weight[n][0]
    elif 83*n25 <= n < 84*n25:
        conv2w83[0][n - 83*n25] = conv2weight[n][0]
    elif 84*n25 <= n < 85*n25:
        conv2w84[0][n - 84*n25] = conv2weight[n][0]
    elif 85*n25 <= n < 86*n25:
        conv2w85[0][n - 85*n25] = conv2weight[n][0]
    elif 86*n25 <= n < 87*n25:
        conv2w86[0][n - 86*n25] = conv2weight[n][0]
    elif 87*n25 <= n < 88*n25:
        conv2w87[0][n - 87*n25] = conv2weight[n][0]
    elif 88*n25 <= n < 89*n25:
        conv2w88[0][n - 88*n25] = conv2weight[n][0]
    elif 89*n25 <= n < 90*n25:
        conv2w89[0][n - 89*n25] = conv2weight[n][0]
    elif 90*n25 <= n < 91*n25:
        conv2w90[0][n - 90*n25] = conv2weight[n][0]
    elif 91*n25 <= n < 92*n25:
        conv2w91[0][n - 91*n25] = conv2weight[n][0]
    elif 92*n25 <= n < 93*n25:
        conv2w92[0][n - 92*n25] = conv2weight[n][0]
    elif 93*n25 <= n < 94*n25:
        conv2w93[0][n - 93*n25] = conv2weight[n][0]
    elif 94*n25 <= n < 95*n25:
        conv2w94[0][n - 94*n25] = conv2weight[n][0]
    elif 95*n25 <= n < 96*n25:
        conv2w95[0][n - 95*n25] = conv2weight[n][0]

# conv2weight mn to nn
conv2w0 = mntonn(conv2w0)
conv2w1 = mntonn(conv2w1)
conv2w2 = mntonn(conv2w2)
conv2w3 = mntonn(conv2w3)
conv2w4 = mntonn(conv2w4)
conv2w5 = mntonn(conv2w5)
conv2w6 = mntonn(conv2w6)
conv2w7 = mntonn(conv2w7)
conv2w8 = mntonn(conv2w8)
conv2w9 = mntonn(conv2w9)
conv2w10 = mntonn(conv2w10)
conv2w11 = mntonn(conv2w11)
conv2w12 = mntonn(conv2w12)
conv2w13 = mntonn(conv2w13)
conv2w14 = mntonn(conv2w14)
conv2w15 = mntonn(conv2w15)
conv2w16 = mntonn(conv2w16)
conv2w17 = mntonn(conv2w17)
conv2w18 = mntonn(conv2w18)
conv2w19 = mntonn(conv2w19)
conv2w20 = mntonn(conv2w20)
conv2w21 = mntonn(conv2w21)
conv2w22 = mntonn(conv2w22)
conv2w23 = mntonn(conv2w23)
conv2w24 = mntonn(conv2w24)
conv2w25 = mntonn(conv2w25)
conv2w26 = mntonn(conv2w26)
conv2w27 = mntonn(conv2w27)
conv2w28 = mntonn(conv2w28)
conv2w29 = mntonn(conv2w29)
conv2w30 = mntonn(conv2w30)
conv2w31 = mntonn(conv2w31)
conv2w32 = mntonn(conv2w32)
conv2w33 = mntonn(conv2w33)
conv2w34 = mntonn(conv2w34)
conv2w35 = mntonn(conv2w35)
conv2w36 = mntonn(conv2w36)
conv2w37 = mntonn(conv2w37)
conv2w38 = mntonn(conv2w38)
conv2w39 = mntonn(conv2w39)
conv2w40 = mntonn(conv2w40)
conv2w41 = mntonn(conv2w41)
conv2w42 = mntonn(conv2w42)
conv2w43 = mntonn(conv2w43)
conv2w44 = mntonn(conv2w44)
conv2w45 = mntonn(conv2w45)
conv2w46 = mntonn(conv2w46)
conv2w47 = mntonn(conv2w47)
conv2w48 = mntonn(conv2w48)
conv2w49 = mntonn(conv2w49)
conv2w50 = mntonn(conv2w50)
conv2w51 = mntonn(conv2w51)
conv2w52 = mntonn(conv2w52)
conv2w53 = mntonn(conv2w53)
conv2w54 = mntonn(conv2w54)
conv2w55 = mntonn(conv2w55)
conv2w56 = mntonn(conv2w56)
conv2w57 = mntonn(conv2w57)
conv2w58 = mntonn(conv2w58)
conv2w59 = mntonn(conv2w59)
conv2w60 = mntonn(conv2w60)
conv2w61 = mntonn(conv2w61)
conv2w62 = mntonn(conv2w62)
conv2w63 = mntonn(conv2w63)
conv2w64 = mntonn(conv2w64)
conv2w65 = mntonn(conv2w65)
conv2w66 = mntonn(conv2w66)
conv2w67 = mntonn(conv2w67)
conv2w68 = mntonn(conv2w68)
conv2w69 = mntonn(conv2w69)
conv2w70 = mntonn(conv2w70)
conv2w71 = mntonn(conv2w71)
conv2w72 = mntonn(conv2w72)
conv2w73 = mntonn(conv2w73)
conv2w74 = mntonn(conv2w74)
conv2w75 = mntonn(conv2w75)
conv2w76 = mntonn(conv2w76)
conv2w77 = mntonn(conv2w77)
conv2w78 = mntonn(conv2w78)
conv2w79 = mntonn(conv2w79)
conv2w80 = mntonn(conv2w80)
conv2w81 = mntonn(conv2w81)
conv2w82 = mntonn(conv2w82)
conv2w83 = mntonn(conv2w83)
conv2w84 = mntonn(conv2w84)
conv2w85 = mntonn(conv2w85)
conv2w86 = mntonn(conv2w86)
conv2w87 = mntonn(conv2w87)
conv2w88 = mntonn(conv2w88)
conv2w89 = mntonn(conv2w89)
conv2w90 = mntonn(conv2w90)
conv2w91 = mntonn(conv2w91)
conv2w92 = mntonn(conv2w92)
conv2w93 = mntonn(conv2w93)
conv2w94 = mntonn(conv2w94)
conv2w95 = mntonn(conv2w95)


##
conv2w0_out = convtest(out0, conv2w0) + convtest(out1, conv2w1) + convtest(out2, conv2w2) + \
    convtest(out3, conv2w3) + convtest(out4, conv2w4) + convtest(out5, conv2w5)

conv2w1_out = convtest(out0, conv2w6) + convtest(out1, conv2w7) + convtest(out2, conv2w8) + \
    convtest(out3, conv2w9) + convtest(out4, conv2w10) + \
    convtest(out5, conv2w11)

conv2w2_out = convtest(out0, conv2w12) + convtest(out1, conv2w13) + convtest(out2, conv2w14) + \
    convtest(out3, conv2w15) + convtest(out4, conv2w16) + \
    convtest(out5, conv2w17)

conv2w3_out = convtest(out0, conv2w18) + convtest(out1, conv2w19) + convtest(out2, conv2w20) + \
    convtest(out3, conv2w21) + convtest(out4, conv2w22) + \
    convtest(out5, conv2w23)

conv2w4_out = convtest(out0, conv2w24) + convtest(out1, conv2w25) + convtest(out2, conv2w26) + \
    convtest(out3, conv2w27) + convtest(out4, conv2w28) + \
    convtest(out5, conv2w29)

conv2w5_out = convtest(out0, conv2w30) + convtest(out1, conv2w31) + convtest(out2, conv2w32) + \
    convtest(out3, conv2w33) + convtest(out4, conv2w34) + \
    convtest(out5, conv2w35)

conv2w6_out = convtest(out0, conv2w36) + convtest(out1, conv2w37)+convtest(out2, conv2w38) + \
    convtest(out3, conv2w39)+convtest(out4, conv2w40)+convtest(out5, conv2w41)
conv2w7_out = convtest(out0, conv2w42) + convtest(out1, conv2w43)+convtest(out2, conv2w44) + \
    convtest(out3, conv2w45)+convtest(out4, conv2w46)+convtest(out5, conv2w47)
conv2w8_out = convtest(out0, conv2w48) + convtest(out1, conv2w49)+convtest(out2, conv2w50) + \
    convtest(out3, conv2w51)+convtest(out4, conv2w52)+convtest(out5, conv2w53)
conv2w9_out = convtest(out0, conv2w54) + convtest(out1, conv2w55)+convtest(out2, conv2w56) + \
    convtest(out3, conv2w57)+convtest(out4, conv2w58)+convtest(out5, conv2w59)
conv2w10_out = convtest(out0, conv2w60) + convtest(out1, conv2w61)+convtest(out2, conv2w62) + \
    convtest(out3, conv2w63)+convtest(out4, conv2w64)+convtest(out5, conv2w65)
conv2w11_out = convtest(out0, conv2w66) + convtest(out1, conv2w67)+convtest(out2, conv2w68) + \
    convtest(out3, conv2w69)+convtest(out4, conv2w70)+convtest(out5, conv2w71)
conv2w12_out = convtest(out0, conv2w72) + convtest(out1, conv2w73)+convtest(out2, conv2w74) + \
    convtest(out3, conv2w75)+convtest(out4, conv2w76)+convtest(out5, conv2w77)
conv2w13_out = convtest(out0, conv2w78) + convtest(out1, conv2w79)+convtest(out2, conv2w80) + \
    convtest(out3, conv2w81)+convtest(out4, conv2w82)+convtest(out5, conv2w83)
conv2w14_out = convtest(out0, conv2w84) + convtest(out1, conv2w85)+convtest(out2, conv2w86) + \
    convtest(out3, conv2w87)+convtest(out4, conv2w88)+convtest(out5, conv2w89)
conv2w15_out = convtest(out0, conv2w90) + convtest(out1, conv2w91)+convtest(out2, conv2w92) + \
    convtest(out3, conv2w93)+convtest(out4, conv2w94)+convtest(out5, conv2w95)


print("conv2 output shape : ", conv2w12_out.shape)

##conv2 Relu ##
conv2w0_out = np.maximum(conv2w0_out, 0)
conv2w1_out = np.maximum(conv2w1_out, 0)
conv2w2_out = np.maximum(conv2w2_out, 0)
conv2w3_out = np.maximum(conv2w3_out, 0)
conv2w4_out = np.maximum(conv2w4_out, 0)
conv2w5_out = np.maximum(conv2w5_out, 0)
conv2w6_out = np.maximum(conv2w6_out, 0)
conv2w7_out = np.maximum(conv2w7_out, 0)
conv2w8_out = np.maximum(conv2w8_out, 0)
conv2w9_out = np.maximum(conv2w9_out, 0)
conv2w10_out = np.maximum(conv2w10_out, 0)
conv2w11_out = np.maximum(conv2w11_out, 0)
conv2w12_out = np.maximum(conv2w12_out, 0)
conv2w13_out = np.maximum(conv2w13_out, 0)
conv2w14_out = np.maximum(conv2w14_out, 0)
conv2w15_out = np.maximum(conv2w15_out, 0)


# conv2 MaxPooling = input fc1
conv2w0_out = pool(conv2w0_out, Maxpooling_matrix)
conv2w1_out = pool(conv2w1_out, Maxpooling_matrix)
conv2w2_out = pool(conv2w2_out, Maxpooling_matrix)
conv2w3_out = pool(conv2w3_out, Maxpooling_matrix)
conv2w4_out = pool(conv2w4_out, Maxpooling_matrix)
conv2w5_out = pool(conv2w5_out, Maxpooling_matrix)
conv2w6_out = pool(conv2w6_out, Maxpooling_matrix)
conv2w7_out = pool(conv2w7_out, Maxpooling_matrix)
conv2w8_out = pool(conv2w8_out, Maxpooling_matrix)
conv2w9_out = pool(conv2w9_out, Maxpooling_matrix)
conv2w10_out = pool(conv2w10_out, Maxpooling_matrix)
conv2w11_out = pool(conv2w11_out, Maxpooling_matrix)
conv2w12_out = pool(conv2w12_out, Maxpooling_matrix)
conv2w13_out = pool(conv2w13_out, Maxpooling_matrix)
conv2w14_out = pool(conv2w14_out, Maxpooling_matrix)
conv2w15_out = pool(conv2w15_out, Maxpooling_matrix)

###conv2 out 量化 ###
conv2w0_out = qq(conv2w0_out * 16*conv2_output_scale)
conv2w1_out = qq(conv2w1_out * 16*conv2_output_scale)
conv2w2_out = qq(conv2w2_out * 16*conv2_output_scale)
conv2w3_out = qq(conv2w3_out * 16 * conv2_output_scale)
conv2w4_out = qq(conv2w4_out * 16*conv2_output_scale)
conv2w5_out = qq(conv2w5_out * 16 * conv2_output_scale)
conv2w6_out = qq(conv2w6_out * 16 * conv2_output_scale)
conv2w7_out = qq(conv2w7_out * 16 * conv2_output_scale)
conv2w8_out = qq(conv2w8_out * 16 * conv2_output_scale)
conv2w9_out = qq(conv2w9_out * 16*conv2_output_scale)
conv2w10_out = qq(conv2w10_out * 16 * conv2_output_scale)
conv2w11_out = qq(conv2w11_out * 16 * conv2_output_scale)
conv2w12_out = qq(conv2w12_out * 16 * conv2_output_scale)
conv2w13_out = qq(conv2w13_out * 16 * conv2_output_scale)
conv2w14_out = qq(conv2w14_out * 16 * conv2_output_scale)
conv2w15_out = qq(conv2w15_out * 16 * conv2_output_scale)


print("second Maxpooling output shape : ", conv2w12_out.shape)

fc1_input = add(conv2w0_out, conv2w1_out)
fc1_input = add(fc1_input, conv2w2_out)
fc1_input = add(fc1_input, conv2w3_out)
fc1_input = add(fc1_input, conv2w4_out)
fc1_input = add(fc1_input, conv2w5_out)
fc1_input = add(fc1_input, conv2w6_out)
fc1_input = add(fc1_input, conv2w7_out)
fc1_input = add(fc1_input, conv2w8_out)
fc1_input = add(fc1_input, conv2w9_out)
fc1_input = add(fc1_input, conv2w10_out)
fc1_input = add(fc1_input, conv2w11_out)
fc1_input = add(fc1_input, conv2w12_out)
fc1_input = add(fc1_input, conv2w13_out)
fc1_input = add(fc1_input, conv2w14_out)
fc1_input = add(fc1_input, conv2w15_out)
print("fc1 input size : ", fc1_input.size)

print("-----for fc1------")
dd = laa(fc1_weight)

x = mulx(fc1_input, 120, dd)
print("fc1 output size : ", x.size)

ReLu_x = np.maximum(x, 0)
ReLu_x = qq(ReLu_x * fc1_output_scale)

print("------for fc2-----")
cc = laa(fc2_weight)
y = mulx(ReLu_x, 84, cc)
print("fc2 output size : ", y.size)

ReLu_y = np.maximum(y, 0)
ReLu_y = qq(ReLu_y * fc2_output_scale)

print("------for fc3-----")
oo = laa(fc3_weight)
z = mulx(ReLu_y, 10, oo)
print("fc3 output size : ", z.size)
z = np.maximum(z, 0)
z = qq(z * fc3_output_scale)
print(z)

print("------fc3.bias-----")
bb = laa(fc3_bias)
print(bb)


print("-------------result = bias + fc3 output----------------")
for n in range(10):
    result = np.zeros((1, 10))
    result[0][n] = float(bb[0][n]) + z[0][n]
    print("result -> ", result[0][n], "=", float(bb[0][n]), "+", z[0][n])
print("-------------result----------------")


print("result = ", result)
print("result shape = ", result.shape, " , ", "result size = ", result.size)


"""
###########算 accuracy###########

max_number = result.max()
pre_label = np.where(result == max_number)

for n in range(10000):
    p , label = trainset[1]
    count = 0
    if label == pre_labe[0][0]l :
        count +=1

acc = count /10000
print("accuracy : ",acc,"%")

"""

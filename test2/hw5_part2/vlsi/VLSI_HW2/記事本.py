###############
###############
###############
#
# 記得查 if__name__=='__main__':
#
###############
###############
###############
import numpy as np
import math
"""
def convtest(a, b):  # a:input b:weight
    j_size = a.shape[0] - b.shape[0] + 1  # output長or寬
    j = np.zeros((j_size, j_size))  # output矩陣
    z = laa(b)  # 拉直b(weight)
    count = j_size - 1  # 2

    for m in range(count):  # 列
        for n in range(count):  # 行
            y = la(a, b, m, n)  # 拉直a要被conv的部分
            for u in range(b.size):  # j矩陣內元素個數(j_size*j_size)
                j[m][n] += y[0][u] * z[0][u]

    return j
"""

"""
正確:

[[27. 35. 43.]
 [51. 59. 67.]
 [30. 38. 46.]]

"""

"""
a = 1 % 3
b = 6 % 3
print(a, b)


out0 = convtest(i_r, conv1w_r0) + convtest(i_g, conv1w_g0) + \
    convtest(i_b, conv1b_0)
out1 = convtest(i_r, conv1w_r1) + convtest(i_g, conv1w_g1) + \
    convtest(i_b, conv1b_1)
out2 = convtest(i_r, conv1w_r2) + convtest(i_g, conv1w_g2) + \
    convtest(i_b, conv1b_2)
out0 = convtest(i_r, conv1w_r3) + convtest(i_g, conv1w_g3) + \
    convtest(i_b, conv1b_3)
out0 = convtest(i_r, conv1w_r4) + convtest(i_g, conv1w_g4) + \
    convtest(i_b, conv1b_4)
out0 = convtest(i_r, conv1w_r5) + convtest(i_g, conv1w_g5) + \
    convtest(i_b, conv1w_b5)


v = np.array([
    [2, 1, 4, 3]
])

gg = 140
g = math.sqrt(gg).is_integer()
if g == 1:
    print(int(math.sqrt(gg)))
else:
    print("fail")
"""


a = np.array([
    [1, 2, 3, 4, 5, 6],
    [4, 5, 6, 7, 4, 6],
    [7, 8, 9, 10, 3, 6],
    [1, 2, 3, 4, 2, 6],
    [4, 3, 2, 1, 1, 6],
    [1, 2, 3, 4, 5, 6]
])

z = np.array([
    [2, 1],
    [2, 3],
])


def lacd(a, b, c, d):  # a:原本矩陣 b:weight c:a的c列 d:a的d行 ->把a拉成b矩陣內元素數的一列
    x = np.zeros((b.shape[0], b.shape[1]))
    for n in range(b.shape[0]):
        for k in range(b.shape[1]):
            x[n][k] = a[n+2*c][k+2*d]

    return x.reshape((1, b.size))


def laa(a):
    x = a.reshape((1, a.size))
    # print(x)
    return x


l = lacd(a, z, 1, 2)
print(l)
print(l.max())


def pool(a, b):
    out_size = int(a.shape[0]/2)  # 3 shape[0]:列
    if (a.shape[0] % 2) == 0.0:
        s = np.zeros((out_size, out_size))  # 3*3
        # print(s)
        z = laa(b)
        count = out_size  # 0 1 2
        for m in range(count):  # 列
            for n in range(count):  # 行
                y = lacd(a, b, m, n)  # 拉直a要被conv的部分
                s[m][n] = y.max()
        return s
    else:
        print("wrong input")


print("---------")
print(pool(a, z))


# conv1 MaxPooling =  input of conv2
out0 = pool(out0, Maxpooling_matrix)
out1 = pool(out1, Maxpooling_matrix)
out2 = pool(out2, Maxpooling_matrix)
out3 = pool(out3, Maxpooling_matrix)
out4 = pool(out4, Maxpooling_matrix)
out5 = pool(out5, Maxpooling_matrix)

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

##conv1 Relu##
out0 = np.maximum(out0, 0)
out1 = np.maximum(out1, 0)
out2 = np.maximum(out2, 0)
out3 = np.maximum(out3, 0)
out4 = np.maximum(out4, 0)
out5 = np.maximum(out5, 0)


def mul(a, b):
    x = np.zeros((20, 20))  # 20*20
    a = laa(a)  # 25
    for n in range(a.shape[1]):  # 0 1 2 3...24re

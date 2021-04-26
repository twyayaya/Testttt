import numpy as np
import math
a = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [4.0, 5.0, 6., 7.0, 4.0],
    [7., 8., 9., 10., 3.],
    [1., 2., 3., 4., 2.],
    [4., 3., 2., 1., 1.]
])
z = np.array([
    [8.6, 1.3],
    [2.5, 3.8],
])

h = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
)

t = np.max(h)
r = np.where(h == t)
print(t)
print(r)
print(r[0][0])
"""
uu = z.round()
print(uu)
print("--")
"""
"""
a[a > 5] = 5
a[a < 3] = 3
print(a)
"""
print("---------")
"""
print(a.max())
#print("type = ", type(a))
print(a.shape)
"""


def clip(a):
    a[a >= 5] = 5
    a[a <= 3] = 3
    return a


print(clip(uu))
print("new---------------")


def qq(a):
    b = a.round()
    b = clip(b)
    return b


print(qq(z))


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


# conv
print("---------")
gg = laa(a)
print(gg)
ggg = np.clip(gg, 2, 5)
print(ggg)
print("---------")

print(mntonn(ggg))


def convtest(a, b):  # a:input b:weight
    if a.size >= b.size:
        j_size = (a.shape[0] - b.shape[0] + 1)  # output長or寬 = 3
        j = np.zeros((j_size, j_size))  # output矩陣 = 3*3
        z = laa(b)  # 拉直b(weight)

        for m in range(j_size):  # 列
            #print("m = ", m)
            for n in range(j_size):  # 行
                #print("n = ", n)
                y = la(a, b, m, n)
            # print("y = ", y)  # 拉直a要被conv的部分
                for u in range(z.size):  # j矩陣內元素個數(j_size*j_size)
                    j[m][n] += y[0][u] * z[0][u]

        return j
    else:
        print("wrong number:a must >= b")


print("------------")
print("start\n")
kk = convtest(a, z)
print(kk)


"""
start

[[27. 35. 43. 39.]
 [51. 59. 67. 47.]
 [30. 38. 46. 37.]
 [21. 19. 17. 15.]]

"""
"""
######test###

kk = convtest(a, z)
print("------------")
print(kk)

# maxpooling
maxn = kk.max()
print("kk = ", maxn)

"""


"""
b = np.zeros((2, 2))
for n in range(2):
    for k in range(2):
        b[n][k] = a[n][k+1]
print(b.reshape(1, 4))

-----------------------------
"""


def test():
    print("HELLO")


input_scale = 0.00784313725490196
conv1_output_scale = 0.0005651640144149312
conv2_output_scale = 0.0019432789408916256
fc1_output_scale = 0.002551974189109988
fc2_output_scale = 0.0028480268876474738
fc3_output_scale = 0.002825382723164684
###################################

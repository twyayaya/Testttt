import csv
import numpy as np
import random

path = 'Wine.csv'
with open(path, newline='') as csvfile:
    rows = csv.reader(csvfile, delimiter=',')
    data = np.asarray(list(rows))
    count = 0


# 開啟輸出的 CSV 檔案
    with open('testing.csv', 'w', newline='') as csvfile, open('training.csv', 'w', newline='') as csvfile2:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        writer2 = csv.writer(csvfile2)

        for n in range(178):

            if(data[n][0] == '1' and count < 1):
                test1 = range(0, 59)
                test2 = random.sample(test1, 59)
                for k in range(0, 59):
                    if k < 18:
                        writer.writerow(data[test2[k]])
                    else:
                        # writer.writerow("------others1")
                        writer2.writerow(data[test2[k]])
                count += 1
            elif(data[n][0] == '2' and count < 2):
                test3 = range(59, 130)
                test4 = random.sample(test3, 71)
                for k in range(0, 71):
                    if k < 18:
                        writer.writerow(data[test4[k]])
                    else:
                        # writer.writerow("------others2")
                        writer2.writerow(data[test4[k]])
                # writer.writerow(data[test4[0]])
                # writer.writerow(data[test4[1]])
                # writer.writerow(data[test4[2]])
                count += 1
            elif(data[n][0] == '3' and count < 3):
                test5 = range(130, 178)
                test6 = random.sample(test5, 48)
                for k in range(0, 48):
                    if k < 18:
                        writer.writerow(data[test6[k]])
                    else:
                        # writer.writerow("------others-------")
                        writer2.writerow(data[test6[k]])
                # writer.writerow(data[test6[0]])
                # writer.writerow(data[test6[1]])
                # writer.writerow(data[test6[2]])
                # print("3 type :")
                # print(data[test6])
                count += 1


print("----------make testing and training , done----------")

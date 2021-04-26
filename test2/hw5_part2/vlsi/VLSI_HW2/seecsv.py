import csv
import numpy as np
import random


with open('input.csv', newline='') as csvfile, open('conv1.weight.csv', newline='') as csvfile2:
    rows = csv.reader(csvfile, delimiter=',')
    inputdata = np.asarray(list(rows))
    print("len of (input.csv) = ", len(inputdata))

    rowss = csv.reader(csvfile2, delimiter=',')
    convweight = np.asarray(list(rowss))
    print("len of (convweight.csv) = ", len(convweight))
    print(convweight)

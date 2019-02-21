import random

import numpy as np

delta = {"1": -1, "2": 1}


def get_train(fname):
    filename = fname
    with open(filename) as fp:
        line = fp.readline().split()
        x = int(line[0])
        n = int(line[1])
        l = int(line[2])
        attr = []
        label = []

        for line in fp:
            temp = line.split()
            temp_list = []
            for i in range(x):
                temp_list.append(float(temp[i]))
            temp_list.append(1)
            attr.append(temp_list)
            label.append(temp[x])
    return n, x, l, attr, label


def get_test(fname):
    filename = fname
    with open(filename) as fp:
        attr = []
        label = []
        for line in fp:
            temp = line.split()
            temp_list = []
            for i in range(x):
                temp_list.append(float(temp[i]))
            temp_list.append(1)
            attr.append(temp_list)
            label.append(temp[x])
    return attr, label


def basic_train(x, l, attr, label):
    weight = [0 for i in range(x + 1)]
    p = 1
    t = 0
    while True:
        missed_sample = []
        missed_sample_class = []
        for i in range(l):
            val = np.matmul(np.asarray(weight).transpose(), np.asarray(attr[i]))
            c = label[i]
            if val*delta[c] >= 0:
                missed_sample.append(attr[i])
                missed_sample_class.append(c)
        if len(missed_sample) == 0:
            break

        for j in range(len(missed_sample)):
            weight = np.asarray(weight) -  delta[missed_sample_class[j]] * np.asarray(missed_sample[j])
        p = (p/(t+1))
        t += 1
    return x, weight


def reward_punishment_train(x, l, attr, label):
    weight = [0 for i in range(x + 1)]
    count = 0
    index = 0
    while count != l:
        val = np.matmul(np.asarray(weight).transpose(), np.asarray(attr[index]))
        c = label[index]
        if val * delta[c] >= 0:
            count = 0
            weight = np.asarray(weight) - (delta[label[index]] * np.asarray(attr[index]))
        else:
            count += 1
        index += 1
        index = index % l
    return x, weight


def pocket_train(x, l, attr, label):
    weight = [0 for i in range(x + 1)]
    p = 1
    t = 0
    ws = []
    hs = 0
    count = 0
    while count <= 100:
        count += 1
        missed_sample = []
        missed_sample_class = []
        h = 0
        for i in range(l):
            val = np.matmul(np.asarray(weight).transpose(), np.asarray(attr[i]))
            c = label[i]
            if val * delta[c] >= 0:
                missed_sample.append(attr[i])
                missed_sample_class.append(c)
            else:
                h += 1
        for j in range(len(missed_sample)):
            weight = np.asarray(weight) - delta[missed_sample_class[j]] * np.asarray(missed_sample[j])
        if h > hs:
            hs = h
            ws.clear()
            for i in weight:
                ws.append(i)
        if len(missed_sample) == 0:
            break
        p = (p / (t + 1))
        t += 1
    return x, ws


def basic_test(weight, x, attr, label):
    count = 0
    for i in range(len(attr)):
        val = np.matmul(np.asarray(weight).transpose(), np.asarray(attr[i]))
        c = label[i]
        if val * delta[c] >= 0:
            continue
        else:
            count += 1

    accuracy = (count/len(attr))*100
    print(accuracy)


def kesler_train(n, x, l, attr, label):
    extended_sample = []
    for i in range(l):
        k = int(label[i]) - 1
        for m in range(n - 1):
            extension = [0 for j in range(n * (x + 1))]
            for o in range(len(attr[i])):
                extension[k * (x+1) + o] = attr[i][o]
            for o in range(len(attr[i])):
                extension[((k+1+m) % n)*(x+1) + o] = -attr[i][o]
            extended_sample.append(extension)
    print(extended_sample)
    weight = [random.randint(0, 1) for j in range(n * (x + 1))]
    print(weight)

    count = 0
    while count <= len(extended_sample):
        for e in extended_sample:
            val = np.matmul(np.asarray(weight).transpose(), np.asarray(e))
            if val < 0:
                weight = np.asarray(weight) + np.asarray(e)
                count = 0
            else:
                count += 1
    print(weight)
    individual_weight = []
    for i in range(n):
        w = []
        for j in range((x+1)):
            w.append(weight[(i*(x+1))+j])
        individual_weight.append(w)
    print(individual_weight)

    return individual_weight


def kesler_test(n, x, l, weight, attr, label):
    count = 0
    for i in range(len(label)):
        tl = int(label[i])
        val = -100000
        lv = 0
        for j in range(n):
            new_val = np.matmul(np.asarray(weight[j]).transpose(), np.asarray(attr[i]))
            if new_val > val:
                val = new_val
                lv = j + 1
        if lv == tl:
            count += 1
    accuracy = (count/len(attr))*100
    print(accuracy)


n, x, l, training_attributes, training_labels = get_train("Train.txt")
weight = kesler_train(n, x, l, training_attributes, training_labels)
test_attributes, test_labels = get_test("Test.txt")
kesler_test(n, x, l, weight, test_attributes, test_labels)


'''
n, x, l, training_attributes, training_labels = get_train("Train.txt")
x, weight = pocket_train(x, l, training_attributes, training_labels)
test_attributes, test_labels = get_test("Test.txt")
basic_test(weight, x, test_attributes, test_labels)
'''


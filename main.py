import random
import numpy as np
import pandas as pd
from collections import deque


def activation(x, source):
    for _ in range(x.shape[0]):
        if x[_, 0] == 0:
            x[_, 0] = source[_, :]
        elif x[_, 0] > 0:
            x[_, 0] = 1
        elif x[_, 0] < 0:
            x[_, 0] = -1
    return x


def prepare_data(file_name: str):
    with open(file_name, mode="r") as f:
        temp_data = []
        temp = []
        for string in f:
            temp += list(map(int, string.strip().split()))
            if string == "\n":
                temp_data.append(temp)
                temp = []
        data = np.array(temp_data)
    return data


def learn(data: np.array):
    w = np.zeros((data.shape[1], data.shape[1]))
    for i in data:
        w += np.matmul((np.matmul(w, i) - i), (np.matmul(w, i) - i).transpose()) / (
            np.matmul(i.transpose(), i) - np.matmul(np.matmul(i.transpose(), w), i)
        )
        for j in range(w.shape[0]):
            w[j, j] = 0
    return w


def train(data: np.array, w: np.array):
    for i in data:
        train = i
        patterns = deque(maxlen=3)
        index = 0
        while True:
            index += 1
            train = activation(np.matmul(w, train), train)
            patterns.append(train)
            print("iteration " + str(index))
            print(train.shape, i.shape, w.shape)
            print(
                (
                    -0.5 * np.matmul(np.matmul(train.transpose(), w), train)
                    - np.matmul(train.transpose(), i)
                )[0][0]
            )
            if len(list(patterns)) == 3:
                if np.array_equal(patterns[0], patterns[2]):
                    break
        yield train.reshape((7, 5))
    print("===============")


if __name__ == "__main__":
    data = prepare_data("learn7x5.txt")
    data = data.reshape((*data.shape, 1))
    train_data = prepare_data("train7x5.txt")
    train_data = train_data.reshape((*train_data.shape, 1))
    weight = learn(data)
    output = train(data, weight)
    train_output = train(train_data, weight)
    """with open("result.txt", "w") as f:
        for sample in output:
            for line in sample:
                f.write(" ".join(list(map(str, list(map(int, line))))) + "\n")
            f.write("\n")
        f.write("=================================================================\n")
        for sample in train_output:
            for line in sample:
                f.write(" ".join(list(map(str, list(map(int, line))))) + "\n")
            f.write("\n")"""
    j = 0
    for sample in data:
        dframe = []
        j += 1
        print("number of vector is ", j)
        a = sample
        for i in range(1, 36):
            indexes = list(range(0, 7 * 5))
            test = np.copy(sample)
            # change input vector by 1, 2, 3.. 35 values
            for _ in range(i):
                index = random.choice(indexes)
                indexes.remove(index)
                test[index] = -test[index]
            result = train(np.array([test]), weight)
            print("count of errors in vector is ", i)
            for _ in result:
                _ = _.reshape(1, 7 * 5)
                out = i
                # dframe.append(out)

        # df = pd.DataFrame(dframe)
        # df.to_csv("csv/result" + str(j) + ".csv")

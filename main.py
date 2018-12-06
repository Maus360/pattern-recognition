import numpy as np


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
        for i in range(100):
            train = activation(np.matmul(w, train), train)
        yield train.reshape((7, 5))


if __name__ == "__main__":
    data = prepare_data("learn7x5.txt")
    data = data.reshape((*data.shape, 1))
    train_data = prepare_data("train7x5.txt")
    train_data = train_data.reshape((*train_data.shape, 1))
    weight = learn(data)
    output = train(data, weight)
    train_output = train(train_data, weight)
    with open("result.txt", "w") as f:
        for sample in output:
            for line in sample:
                f.write(" ".join(list(map(str, list(map(int, line))))) + "\n")
            f.write("\n")
        f.write("=================================================================\n")
        for sample in train_output:
            for line in sample:
                f.write(" ".join(list(map(str, list(map(int, line))))) + "\n")
            f.write("\n")

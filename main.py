import numpy as np

def activation(x, source):
    print(source.shape)
    for _ in range(x.shape[1]):
        if x[:, _] == 0:
            x[:, _] = source[_, :]
        elif x[:, _] > 0:
            x[:, _] = 1
        elif x[:, _] < 0:
            x[:, _] = -1
    return x

def prepare_data(file_name:str):
    with open(file_name, mode="r") as f:
        temp_data = []
        temp = []
        for string in f:
            temp += list(map(int, string.strip().split()))
            if string == '\n':
                temp_data.append(temp)
                temp = []
        data = np.array(temp_data)
    return data

def learn(data:np.array):
    w = np.matmul(data[0], data[0].transpose())
    for x in data[1:]:
        w += np.matmul(x, x.transpose())
    w = (w / data.shape[0]) - np.ones(w.shape)
    for i in data:
        s = np.matmul(i.transpose(), w)
        print(activation(s, data[0]))





if __name__=="__main__":
    data = prepare_data("test.txt")
    data = data.reshape((*data.shape, 1))
    output = learn(data)
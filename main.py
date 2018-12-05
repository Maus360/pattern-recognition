import numpy as np

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
    pass




if __name__=="__main__":
    data = prepare_data("test.txt")
    output = learn(data)
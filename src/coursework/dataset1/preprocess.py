import numpy as np
from random import shuffle


class PreprocessingDataSet:
    def __init__(self, train_inp, train_out, test_inp, test_out):
        self.train_inp = train_inp
        self.train_out = train_out
        self.test_inp = test_inp
        self.test_out = test_out


def prepare():

    # Dataset

    dataset1 = [
        "00000 0",
        "00001 0",
        "00010 0",
        "00011 1",
        "00100 0",
        "00101 1",
        "00110 1",
        "00111 0",
        "01000 0",
        "01001 1",
        "01010 1",
        "01011 0",
        "01100 1",
        "01101 0",
        "01110 0",
        "01111 1",
        "10000 0",
        "10001 1",
        "10010 1",
        "10011 0",
        "10100 1",
        "10101 0",
        "10110 0",
        "10111 1",
        "11000 1",
        "11001 0",
        "11010 0",
        "11011 1",
        "11100 0",
        "11101 1",
        "11110 1",
        "11111 0",
    ]

    # shuffle
    shuffle(dataset1)

    train = int(len(dataset1) * 0.75)
    test = len(dataset1) - train

    # Splitting features and labels
    inp = []
    out = []
    for data in dataset1:
        features, label = data.split()
        inp.append([int(x) for x in features])
        out.append(int(label))

    # setting up ndarrays
    train_inp = np.array(inp[:train])
    train_out = np.array(out[:train])
    test_inp = np.array(inp[train:])
    test_out = np.array(out[train:])

    # test a bit
    assert test == len(test_out)

    return PreprocessingDataSet(
        train_inp=train_inp,
        train_out=train_out,
        test_inp=test_inp,
        test_out=test_out,
    )

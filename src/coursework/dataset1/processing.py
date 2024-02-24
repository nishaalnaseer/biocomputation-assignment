from preprocess import PreprocessingDataSet
from sklearn.neural_network import MLPClassifier


def process(dataset: PreprocessingDataSet, rate, _class):
    clf = _class(
        hidden_layer_sizes=(8, 8),
        activation='relu',
        solver='adam',
        max_iter=10000000,
        random_state=42,
        learning_rate='adaptive',
        learning_rate_init=rate,
    )

    clf.fit(dataset.train_inp, dataset.train_out)
    accuracy = clf.score(dataset.test_inp, dataset.test_out)
    return accuracy

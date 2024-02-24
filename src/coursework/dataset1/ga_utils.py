import random
from copy import deepcopy
from typing import List, Tuple
import math
from preprocess import PreprocessingDataSet, prepare
import numpy as np


def create_individual(allele_count: int) -> List[float]:
    """
    Create an individual with n number of alleles
    :param allele_count: number of alleles to generate
    :return:
    """
    return [random.randint(-100, 100) / 100 for _ in range(allele_count)]


def create_population(size: int, length: int) -> List[List[float]]:
    """
    Create an initial population
    :param size: the size of the population
    :param length: the number of alleles each individual has
    :return:
    """
    return [create_individual(length) for _ in range(size)]


def calculate_fitness(individual, dataset: PreprocessingDataSet, sums) -> float:
    """
    calculate the fitness of an individual over a dataset
    :param individual: the individual to be questioned
    :param dataset: the dataset to be tested / trained
    :return:
    """
    _sum = 0
    for data in dataset.train_inp:
        for index, point in enumerate(individual):
            dataset_point = data[index]

            if dataset.train_out[index] == 0:
                out = -1
            else:
                out = 1

            _sum += dataset_point * point * out
            sums.append(_sum)

    return _sum


def train_select_best(individuals, dataset: PreprocessingDataSet, best_count: int, sums: List[float]):
    """
    Calculate the fitness of individuals for a dataset and select the best n number of individuals
    :param sums: the fitness value for all individuals to be normalised
    :param individuals: our population
    :param dataset: our dataset
    :param best_count: the top n number of individuals to selected
    :return:
    """
    trained = {
        index: calculate_fitness(individual, dataset, sums) for index, individual in enumerate(individuals)
    }

    sorted_trained = {k: v for k, v in sorted(trained.items(), key=lambda item: item[1], reverse=True)}
    return [individuals[x] for x in sorted_trained][:best_count]


def mutate(individual, mutation_rate):
    rate_check = random.randint(0, 100)
    if rate_check < mutation_rate:
        for index, allele in enumerate(individual):
            diff = random.randint(-5, 5) * 0.01

            new_val = allele + diff

            if new_val > 1:
                new_val = 1
            elif new_val < -1:
                new_val = -1

            individual[index] = new_val

    return individual


def genetic_recombination(
        allele_positions: List[int],
        parent1: List[float],
        parent2: List[float],
) -> tuple[List[float], List[float],]:
    """
    Genetic recombination of selected alleles between two parents
    :param allele_positions: the positions of the parents to be swapped
    :param parent1: lists
    :param parent2: lists
    :return: children
    """

    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)

    # child1 = mutate(_parent1, 5)
    # child2 = mutate(_parent2, 5)

    for allele_position in allele_positions:
        swap_holder = child1[allele_position]
        child1[allele_position] = child2[allele_position]
        child2[allele_position] = swap_holder

    return child1, child2,


def breed(population):
    length = len(population[0])
    indexes = [x for x in range(length)]
    children = []
    for index1, parent1 in enumerate(population):
        for index2 in range(index1+1, len(population)):
            parent2 = population[index2]

            is_odd = length % 2
            swap_another = random.randint(0, is_odd)

            if swap_another:
                val = math.ceil(length / 2)
            else:
                val = math.floor(length / 2)

            swap_points = random.sample(indexes, val)

            child1, child2 = genetic_recombination(swap_points, parent1, parent2)

            children.append(child1)
            children.append(child2)

    return children


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def test(optimal: List[float], dataset: PreprocessingDataSet, _min, _max):
    right = 0
    wrong = 0
    for index1, _input in enumerate(dataset.test_inp):
        _sum = 0

        for index, point in enumerate(_input):
            _sum += optimal[index] * point

        normal = ((_sum - _max) / (_max - _min)) * 2 - 1
        actual_out = dataset.test_out[index1]

        if normal > 0:
            predicted = 1
        else:
            predicted = 0

        if predicted == actual_out:
            right += 1
        else:
            wrong += 1

    return right, wrong


def normalise(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [((x - min_val) / (max_val - min_val)) * 2 - 1 for x in data]
    return normalized_data


def main():
    dataset = prepare()
    population = create_population(1000, 5)
    _children_holder = []
    sums = []

    for _ in range(100):
        best = train_select_best(population, dataset, 10, sums)

        for child in best:
            _children_holder.append(child)

        population = breed(best)

    final_run = train_select_best(population, dataset, 10, sums)
    for child in final_run:
        _children_holder.append(child)

    final = train_select_best(_children_holder, dataset, 1, sums)

    _min = min(sums)
    _max = max(sums)

    best = final[0]

    right, wrong = test(best, dataset, _min, _max)

    print(right, wrong)


if __name__ == '__main__':
    main()

import asyncio
import random
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime
from typing import List, Tuple
import math
from preprocess import PreprocessingDataSet, prepare
import numpy as np


def create_population(size: int, allele_count: int) -> np.ndarray:
    """
    Create an initial population
    :param size: the size of the population
    :param allele_count: the number of alleles each individual has
    :return:
    """
    return np.random.uniform(0, 1, size=(size, allele_count))


def calculate_fitness(population: np.ndarray, dataset: PreprocessingDataSet) -> float:
    """
    calculate the fitness of an individual over a dataset
    :param population: the population to be questioned
    :param dataset: the dataset to be tested / trained
    :return:
    """
    return np.dot(dataset.train_inp, population.T).sum(axis=1)


def train_select_best(population, dataset: PreprocessingDataSet, best_count: int):
    """
    Calculate the fitness of individuals for a dataset and select the best n number of individuals
    :param population: our population
    :param dataset: our dataset
    :param best_count: the top n number of individuals to selected
    :return:
    """
    stamp = datetime.now().strftime("%Y/%b/%d-%H:%M:%S")
    fitness = calculate_fitness(population, dataset)

    # print(f"{stamp} Fitness length = {len(fitness)}")
    # print(f"{stamp} Population length = {len(population)}")

    try:
        p = population[np.argsort(fitness)]
    except IndexError as e:
        print(f"{stamp} {e}")
        raise Exception("Lel")

    return p


def mutate(individual, mutation_rate):
    rate_check = random.randint(0, 100)

    if rate_check < mutation_rate:
        indexes = [x for x in range(len(individual))]

        randomised = random.sample(indexes, len(individual))

        for index in randomised:
            allele = individual[index]
            randomized = random.randint(-100, 100) * 0.01
            diff = allele - randomized

            new_val = diff / 2

            individual[index] = new_val

    return individual


def genetic_recombination(parent1, parent2):
    """
    Perform genetic recombination (crossover) between two parent individuals.

    Arguments:
    parent1: First parent individual (numpy array).
    parent2: Second parent individual (numpy array).

    Returns:
    child: Resulting child individual after crossover (numpy array).
    """
    # Assuming simple crossover by randomly selecting crossover point
    crossover_point = random.randint(0, len(parent1))
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    return child


def breed(population, mutation_rate):
    children = deepcopy(population)
    mutation_prob = mutation_rate / 100

    for child in children:
        if random.random() < mutation_prob:
            child += np.random.uniform(-0.1, 0.1, size=child.shape)

    np.clip(children, 0, 1, out=children)

    num_children = len(children)
    num_parents = len(population)
    array = []

    for _ in range(num_children):
        parent1, parent2 = random.sample(range(num_parents), 2)
        child = genetic_recombination(children[parent1], children[parent2])
        array.append(child)

    return np.array(array)


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


def test(optimal: np.ndarray, dataset: PreprocessingDataSet) -> Tuple[int, int]:
    preds = np.dot(dataset.test_inp, optimal)
    preds = sigmoid(preds)  # Applying sigmoid activation
    preds = np.where(preds > 0.5, 1, 0)  # Thresholding at 0.5
    correct = np.sum(preds == dataset.test_out)  # Counting correct predictions
    incorrect = len(dataset.test_out) - correct  # Total - correct = incorrect predictions

    # swapped this because as per statistics incorrect predictions are much more likely
    return incorrect, correct


def normalise(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [((x - min_val) / (max_val - min_val)) * 2 - 1 for x in data]
    return normalized_data


def run_ga(mutation_rate: int, iteration: int):
    dataset = prepare()
    population = create_population(1000, 5)
    _children_holder = []

    for _ in range(2000):
        best = train_select_best(population, dataset, 10)
        children = breed(best, mutation_rate)
        population = np.vstack((best, children))

    final_run = train_select_best(population, dataset, 1)[0]
    right, wrong = test(final_run, dataset)
    print(f"Iteration: {iteration}, Mutation rate: {mutation_rate}%, Right: {right}, Wrong: {wrong}")


async def main(iterations: int):
    with ProcessPoolExecutor(max_workers=6) as executor:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, run_ga, *[0, _+1]) for _ in range(iterations)]
        await asyncio.gather(*tasks)

    print("\n\n\n")

    with ProcessPoolExecutor(max_workers=6) as executor:
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, run_ga, *[30, _+1]) for _ in range(iterations)]
        await asyncio.gather(*tasks)


if __name__ == '__main__':
    asyncio.run(main(100))

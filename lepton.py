# TODO: Finish building the models
#       Figure out how to distribute the datasets (potentially using .pop or making some sort of stack)
#       Figure out why checker function isn't working
#       Build function that pops the stack and distributes the dataset to each selected algorithm

import csv
import pandas as pd
import numpy as np
import itertools
import math
import random

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# "DATA PREPROCESSING"
# # Open instances (X_data)
# split_data = []

# with open(r"Neutrinos.csv", "r") as file:
#     X_reader = csv.reader(file, delimiter = "\t")
#     for line in X_reader:
#         split_data.append([string.split(",") for string in line])

# # Removing redundant list of list
# neutrinos = []

# for listed_list in split_data:
#     for neutrino in listed_list:
#         neutrinos.append(neutrino)

# # Open labels (Y_data)
# with open(r"Neutrinos labels.csv", "r") as file:
#     reader = csv.reader(file, delimiter = "\t")
#     reader = list(reader)

# # Assing variable to Y_data
# Y_data = [item for sublist in reader for item in sublist]

# # Find max and min of all the instances to normalize
# raw_instances = []

# for neutrino in neutrinos:
#     for instance in neutrino:
#         raw_instances.append(float(instance))

# top = float(max(raw_instances))
# bottom = float(min(raw_instances))

# assert len(neutrinos) == len(Y_data), "Datasets are not the same size"

# # Normalize X_data
# norm_neutrinos = []

# def normalizer(datasets):
#     for neutrino in neutrinos:
#         norm_neutrinos.append([(float(instance) - bottom) / (top - bottom) for instance in neutrino])

# normalizer(neutrinos)

# """WRITTING NEW CSV FILE"""
# # Writting normalized dataset into a new file using csv
# def writeCsvFile(fname, data, *args, **kwargs):
#     mycsv = csv.writer(open(fname, 'w'), *args, **kwargs)
#     for row in norm_neutrinos:
#         mycsv.writerow(row)

# writeCsvFile(r'neutrino_data_norm.csv', norm_neutrinos)

"""OPEN FORMATTED DATASET"""
formatted_X_data = pd.read_csv(r"neutrino_data_norm.csv")
formatted_Y_data = pd.read_csv(r"Neutrinos labels.csv")

dataset_size = len(formatted_X_data)
input_size = len(formatted_X_data.columns)

"""CHOICE FUNCTIONS"""

possible_algorithms = ["ANN", "RF", "KNN", "SVM", "GPC", "GNB", "BNB"]

# Define your variables here
X_dataset = [list(x) for x in formatted_X_data.values]
Y_dataset = [int(y) for y in formatted_Y_data.values]

assert len(X_dataset) == len(Y_dataset)

train_split = .7
test_split = .3

X_data_len = len(X_dataset)
Y_data_len = len(Y_dataset)

# Data splitter, receives number of algorithms and the dataset as inputs, and outputs the appropriate number of datasets
def shuffler(inputs, targets):
    combined = list(zip(inputs, targets))
    shuffled = shuffle(combined)
    X, Y = zip(*shuffled)

    return X, Y

X_data, Y_data = shuffler(X_dataset, Y_dataset)

datasets = [X_data, Y_data]

def splitter(datasets):
    processed_datasets = []

    def user_choices():
        algorithms = []

        choices = input("Input the algorithms you would like to use. You have the following choices: ANN, RF, KNN, SVM, GPC, GNB, BNB \n",)

        models = choices.split(", ")

        for model in tqdm(models):
            model = model.upper()
            if model in possible_algorithms:
                algorithms.append(model)

        return algorithms

    len_of_algorithms = len(user_choices())

    for dataset in datasets:
        if len_of_algorithms == 1:
            processed_datasets = train_test_split(dataset, test_size=test_split, train_size=.7, shuffle=False)

            return processed_datasets
        else:
            split = math.floor(dataset_size / len_of_algorithms) # Rounded down
            split_data = [dataset[x:x+split] for x in range(0, len(dataset), split)]
            split_data.pop()[-1]

        for unique_dataset in split_data:
            divided_dataset = train_test_split(unique_dataset, test_size=test_split, train_size=.7, shuffle=False)
            processed_datasets.append(divided_dataset)

    return processed_datasets

processed_datasets = splitter(datasets)

def seperator(processed_datasets):
    half = len(processed_datasets)//2
    return processed_datasets[:half], processed_datasets[half:]

X_data_pairs, Y_data_pairs = seperator(processed_datasets)

def checker(final_X_datasets, final_Y_datasets):
    for X_train_test_pairs in final_X_datasets:
        X_len = len(X_train_test_pairs)
        return X_len

    for Y_train_test_pairs in final_X_datasets:
        Y_len = len(Y_train_test_pairs)
        return Y_len

    if X_len == Y_len:
        print("Dataset size and format confirmed and matched; Proceed")

    else:
        print("Dataset size and format failed and denied; Revise")

checker(X_data_pairs, Y_data_pairs)

for thing in processed_datasets:
    print(len(thing)) # print "2" 6 times
    for ting in thing:
        print(len(ting)) # prints 70% and 30% of the dataset 6 times

# X_data_train, X_data_test = splitter(algorithms, X_dataset)
# Y_data_train, Y_data_test = splitter(algorithms, Y_dataset)

# print(len(X_data_test), len(X_data_train))

# print(len(X_data))

# def divider(datasets):
#     for dataset in datasets:
#         divided_dataset = train_test_split(dataset, test_size=test_split, train_size=.7, shuffle=True)

#         return divided_dataset

# X_data = divider(split_X_data)
# Y_data = divider(split_Y_data)

# print(len(split_X_data[2]))
# for dataset in X_data:
#     print(len(dataset))
# print(formatted_X_data.sample())
# print(chance)
# def choose_type_algo(number_of_algorithms, type_of_algorithms, split_dataset):
    

# def choose(dataset, number_of_algorithms, algorithms):

"""YEET"""

# X_data_len = len(X_dataset)
# Y_data_len = len(Y_dataset)

# # Data splitter, receives number of algorithms and the dataset as inputs, and outputs the appropriate number of datasets
# def splitter(algorithms, X_dataset, Y_dataset):
#     processed_X_datasets = []
#     processed_Y_datasets = []

#     if len(algorithms) == 1:
#         X_dataset = train_test_split(X_dataset, test_size=test_split, train_size=.7, shuffle=True)
#         Y_dataset = train_test_split(Y_dataset, test_size=test_split, train_size=.7, shuffle=True)
#         return X_dataset, Y_dataset
#     else:
#         split = math.floor(dataset_size / len(algorithms)) # Rounded down

#         split_X_data = [X_dataset[x:x+split] for x in range(0, X_data_len, split)]
#         split_Y_data = [Y_dataset[x:x+split] for x in range(0, Y_data_len, split)]

#         split_X_data.pop()[-1]
#         split_Y_data.pop()[-1]

#         for unique_dataset in split_data:
#             divided_X_dataset = train_test_split(unique_X_dataset, test_size=test_split, train_size=.7, shuffle=True)
#             divided_Y_dataset = train_test_split(unique_Y_dataset, test_size=test_split, train_size=.7, shuffle=True)

#             processed_X_datasets.append(divided_X_dataset)
#             processed_Y_datasets.append(divided_Y_dataset)

#             return processed_X_datasets, processed_Y_datasets

# processed_X_datasets = splitter(user_choices(), X_dataset)
# processed_Y_datasets = splitter(user_choices(), Y_dataset)
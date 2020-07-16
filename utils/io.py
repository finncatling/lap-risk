import pickle
import os


def save_object(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, "rb") as input_file:
        return pickle.load(input_file)


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

from code.base_class.dataset import dataset
import os
import string
from nltk import word_tokenize
import pickle


class Dataset_Loader(dataset):
    dataset_source_folder_path = None
    train_pos_source_file_name = None
    train_neg_source_file_name = None
    test_pos_source_file_name = None
    test_neg_source_file_name = None

    def load(self):
        print("loading data...")

        neg_train = []
        pos_train = []
        neg_test = []
        pos_test = []

        with open(
            self.dataset_source_folder_path + self.train_pos_source_file_name, "rb"
        ) as file:
            pos_train = pickle.load(file)

        with open(
            self.dataset_source_folder_path + self.train_neg_source_file_name, "rb"
        ) as file:
            neg_train = pickle.load(file)

        with open(
            self.dataset_source_folder_path + self.test_pos_source_file_name, "rb"
        ) as file:
            pos_test = pickle.load(file)

        with open(
            self.dataset_source_folder_path + self.test_neg_source_file_name, "rb"
        ) as file:
            neg_test = pickle.load(file)

        data = {
            "train": {
                "X": neg_train + pos_train,
                "y": [0] * len(neg_train) + [1] * len(pos_train),
            },
            "test": {
                "X": neg_test + pos_test,
                "y": [0] * len(neg_test) + [1] * len(pos_test),
            },
        }
        print(data["test"]["X"][0])
        print(data["test"]["y"][0])

        return data

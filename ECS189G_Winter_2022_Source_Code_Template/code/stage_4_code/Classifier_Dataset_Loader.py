from io import TextIOWrapper
from code.base_class.dataset import dataset
import os
from nltk import word_tokenize


class Dataset_Loader(dataset):
    train_data = None
    test_data =None
    train_labels = None
    test_labels = None
    test_dataset_source_folder_path = None
    train_dataset_source_folder_path = None
    dataset_source_file_name = None
    cmap = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.data = []
        self.labels = []

    def load_data(self, directory, list: list):
        for file in os.listdir(directory):
            with open(os.path.join(directory, file), 'rt', encoding='utf-8') as f:
                line = f.readline()
                line = line.strip('\n')
                tokens = word_tokenize(line)
                list.append(tokens)
    def load(self):
        print("loading data...")

        train_pos_path = (
            self.dataset_source_folder_path + self.traindata_pos_source_file_name
        )
        train_neg_path = (
            self.dataset_source_folder_path + self.traindata_neg_source_file_name
        )
        test_pos_path = (
            self.dataset_source_folder_path + self.testdata_pos_source_file_name
        )
        test_neg_path = (
            self.dataset_source_folder_path + self.testdata_neg_source_file_name
        )

        neg_train = []
        pos_train = []
        self.load_data(train_neg_path, neg_train)
        self.load_data(train_pos_path, pos_train)

        neg_test = []
        pos_test = []
        self.load_data(test_neg_path, neg_test)
        self.load_data(test_pos_path, pos_test)

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

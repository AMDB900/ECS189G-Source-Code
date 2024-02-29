from io import TextIOWrapper
from code.base_class.dataset import dataset
import os
import nltk



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

    def load_data(self, data, labels, directory, label):
        for file in os.listdir(directory):
            with open(os.path.join(directory, file), 'rt') as f:
                for line in f:
                    line = line.strip('\n')
                    tokens = word_tokenize(line)
                    data.append(tokens)
                labels.append(label)
                f.close()

    def append_tokens(self, f: TextIOWrapper, list: list):
        line = f.readline()
        line = line.strip("\n")
        tokens = nltk.word_tokenize(line)
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

        for file in os.listdir(train_pos_path):
            with open(train_pos_path + file, "r", encoding="utf8") as f:
                self.append_tokens(f, pos_train)

        for file in os.listdir(train_neg_path):
            with open(train_neg_path + file, "r", encoding="utf8") as f:
                self.append_tokens(f, neg_train)

        neg_test = []
        pos_test = []
        for file in os.listdir(test_pos_path):
            with open(test_pos_path + file, "r", encoding="utf8") as f:
                self.append_tokens(f, pos_test)

        for file in os.listdir(test_neg_path):
            with open(test_neg_path + file, "r", encoding="utf8") as f:
                self.append_tokens(f, neg_test)

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

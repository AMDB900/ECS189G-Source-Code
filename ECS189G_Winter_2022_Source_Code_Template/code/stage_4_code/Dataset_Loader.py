from code.base_class.dataset import dataset
import pickle
import os
import nltk
from nltk.tokenize import word_tokenize


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
                data.append(f.read())
                labels.append(label)
                f.close()

    def load(self):
        self.load_data(self.train_data, self.train_labels, self.train_dataset_source_folder_path + "pos", 1)
        self.load_data(self.train_data, self.train_labels, self.train_dataset_source_folder_path + "neg", 0)
        self.load_data(self.test_data, self.test_labels, self.test_dataset_source_folder_path + "pos", 1)
        self.load_data(self.test_data, self.test_labels, self.test_dataset_source_folder_path + "neg", 0)

        return {
            'train': {
                'text': [self.train_data],
                'label': [self.train_labels]
            },
            'test': {
                'text': [self.test_data],
                'label': [self.test_labels]
            }
        }

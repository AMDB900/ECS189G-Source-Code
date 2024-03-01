'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
from nltk import word_tokenize

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    data_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        X_train = []

        with open(self.dataset_source_folder_path + self.data_source_file_name, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                tokens = word_tokenize(line)
                X_train.append(tokens)

        return {'train': {'X': X_train}}
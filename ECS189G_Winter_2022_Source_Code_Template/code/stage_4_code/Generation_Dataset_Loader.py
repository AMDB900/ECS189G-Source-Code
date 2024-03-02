"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
from nltk import word_tokenize
import csv

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    data_source_file_name = None

    num_inputs = 3
    num_outputs = 1

    def load(self):
        print('loading data...')
        X_train = []
        y_train = []
        X_test = []

        with open(
            self.dataset_source_folder_path + self.data_source_file_name,
            "r",
            encoding="utf-8",
        ) as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == "ID":
                    continue
                tokens = word_tokenize(row[1])
                tokens.append("ENDTOKEN")
                for i in range(0, len(tokens) - self.num_inputs):
                    X_train.append(tokens[i : i + self.num_inputs])
                    y_train.append(
                        tokens[
                            i + self.num_inputs : i + self.num_inputs + self.num_outputs
                        ]
                    )
                    if i == 0:
                        X_test.append(tokens[i : i + self.num_inputs])

        return {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test}}

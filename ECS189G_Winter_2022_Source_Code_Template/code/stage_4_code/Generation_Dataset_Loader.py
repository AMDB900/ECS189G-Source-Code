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

    num_in = 3
    num_out = 1
    bad_tokens = [
        "*",
        "-",
        "^",
        "=",
        '"',
        "[",
        "]",
        "/",
        "``",
        "''",
        "&",
        "'",
        "(",
        ")",
        "..",
        "...",
        "....",
        ".....",
    ]

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
                tokens = [token for token in tokens if token not in self.bad_tokens]
                tokens.append("ENDTOKEN")
                for i in range(0, len(tokens) - self.num_in):
                    X_train.append(tuple(tokens[i : i + self.num_in]))
                    y_train.append(
                        tuple(tokens[i + self.num_in : i + self.num_in + self.num_out])
                    )
                    if i == 0:
                        X_test.append(tuple(tokens[i : i + self.num_in]))

        X_test = list(set(X_test))

        return {"train": {"X": X_train, "y": y_train}, "test": {"X": X_test}}

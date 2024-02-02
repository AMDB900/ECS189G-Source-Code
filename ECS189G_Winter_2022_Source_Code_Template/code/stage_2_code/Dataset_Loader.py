'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    traindata_source_file_name = None
    testdata_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X_train = []
        y_train = []
    
        with open(self.dataset_source_folder_path + self.traindata_source_file_name, 'r') as f:
            for line in f:
                line = line.strip('\n')
                elements = [int(i) for i in line.split(',')]
                X_train.append(elements[1:])
                y_train.append(elements[0])

        X_test = []
        y_test = []
        with open(self.dataset_source_folder_path + self.testdata_source_file_name, 'r') as f:
            for line in f:
                line = line.strip('\n')
                elements = [int(i) for i in line.split(',')]
                X_test.append(elements[1:])
                y_test.append(elements[0])
                
        return {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
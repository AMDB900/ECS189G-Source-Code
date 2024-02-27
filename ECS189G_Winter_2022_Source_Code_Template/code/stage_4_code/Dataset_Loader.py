from code.base_class.dataset import dataset
import nltk

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None

    traindata_pos_source_file_name = None
    traindata_neg_source_file_name = None

    testdata_pos_source_file_name = None
    testdata_neg_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        neg_train = []
        pos_train = []

        with open(self.dataset_source_folder_path + self.traindata_pos_source_file_name, 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = nltk.word_tokenize(line)
                pos_train.append(tokens)

        with open(self.dataset_source_folder_path + self.traindata_neg_source_file_name, 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = nltk.word_tokenize(line)
                neg_train.append(tokens)

        neg_test = []
        pos_test = []
        with open(self.dataset_source_folder_path + self.testdata_pos_source_file_name, 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = nltk.word_tokenize(line)
                pos_test.append(tokens)

        with open(self.dataset_source_folder_path + self.testdata_neg_source_file_name, 'r') as f:
            for line in f:
                line = line.strip('\n')
                tokens = nltk.word_tokenize(line)
                neg_test.append(tokens)

        return {'train': {'Pos': pos_train, 'Neg': neg_train}, 'test': {'Pos': pos_test, 'Neg': neg_test}}
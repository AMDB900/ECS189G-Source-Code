'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')

        accuracy = accuracy_score(self.data['true_y'], self.data['pred_y'])
        precision, recall, fscore, support = precision_recall_fscore_support(self.data['true_y'], self.data['pred_y'], average='weighted')

        return accuracy, precision, recall, fscore
        
"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from torch.utils.data import TensorDataset, DataLoader
from code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import time
from gensim.models import Word2Vec
import gensim

class Method_Classification(method, nn.Module):
    data = None
    glove_embeddings = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 5000
    max_epoch = 1200
    learning_rate = 1e-3

    hidden_size = 50
    num_layers = 1

    loss_history = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.rnn = nn.RNN(50, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, 2)
        self.dropout = nn.Dropout(0.05)
        self.glove_embeddings = self.load_glove("data/stage_4_data/glove.6B.50d.txt")
        self.gensim_model = None
    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, X):
        out, _ = self.rnn(X)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out
    def load_glove(self, glove_file):
        index = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefficients = np.asarray(values[1:], dtype='float32')
                index[word] = coefficients
        return index
    # Matches words in the vectors to glove embeddings
    def data_preprocess(self, X):
        embedding_length = 50
        max_review_length = 50
        tensor = np.zeros((len(X), max_review_length, embedding_length))

        for i, review in enumerate(X):
            for j, word in enumerate(review):
                if j == max_review_length:
                    break
                if word not in self.glove_embeddings:
                    continue
                embedding = self.glove_embeddings.get(word)

                tensor[i, j, :] = embedding

        return tensor

    def data_embed(self, X):
        embedding_length = 50
        max_review_length = 50
        tensor = np.zeros((len(X), max_review_length, embedding_length))
        unk_embedding = np.ones(embedding_length, dtype='float32')
        for i, review in enumerate(X):
            for j, word in enumerate(review):
                if j == max_review_length:
                    break
                embedding = self.gensim_model.wv[word] if word in self.gensim_model.wv else unk_embedding

                tensor[i, j, :] = embedding
        return tensor

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=5e-3)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy("training evaluator", "")
        dataset = TensorDataset(
            torch.tensor(self.data_embed(X), device=self.device, dtype=torch.float32),
            torch.tensor(np.array(y), device=self.device, dtype=torch.long),
        )
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.max_epoch):
            running_loss = 0.0
            for inputs, y_true in train_loader:
                optimizer.zero_grad()
                y_pred = self.forward(inputs)
                train_loss = loss_function(y_pred, y_true)
                train_loss.backward()
                optimizer.step()
                running_loss += train_loss.item()

            self.loss_history.append(running_loss / len(train_loader))
            if epoch % 10 == 0 or epoch == self.max_epoch - 1:
                accuracy_evaluator.data = {
                    "true_y": y_true.cpu(),
                    "pred_y": y_pred.cpu().max(1)[1],
                }
                accuracy = accuracy_evaluator.evaluate()
                print(
                    "Epoch:",
                    epoch,
                    "Accuracy:",
                    accuracy,
                    "Loss:",
                    running_loss / len(train_loader),
                )

    def test(self, X):
        test_loader = DataLoader(torch.tensor(self.data_embed(X), device=self.device, dtype=torch.float32),
                                 batch_size=250)

        y_pred_list = []

        for inputs in test_loader:
            outputs = self.forward(inputs)
            y_pred_list.append(outputs.max(1)[1])

        y_pred = torch.cat(y_pred_list)

        return y_pred

    def run(self):
        start = time.perf_counter()
        print("method running...")
        print("--start training...")
        data = self.data["train"]["X"]
        self.gensim_model = gensim.models.Word2Vec(data, min_count=1, vector_size=50, window=5)
        self.train(self.data["train"]["X"], self.data["train"]["y"])
        print("--start testing...")
        pred_y_train = self.test(self.data["train"]["X"])
        pred_y_test = self.test(self.data["test"]["X"])
        end = time.perf_counter()
        print((end - start)/60, " minutes elapsed")
        return (
            {
                "pred_y": pred_y_train.cpu(),
                "true_y": self.data["train"]["y"],
            },
            {
                "pred_y": pred_y_test.cpu(),
                "true_y": self.data["test"]["y"],
            },
            self.loss_history,
        )

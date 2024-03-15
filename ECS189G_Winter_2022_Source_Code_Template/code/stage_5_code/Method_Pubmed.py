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
import torch.nn.functional as F
from code.stage_5_code.GraphConvolution import GraphConvolution


class Method_Pubmed(method, nn.Module):
    data = None
    adj = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training = False

    max_epoch = 200
    learning_rate = 1e-3

    hidden_size = 50
    dropout = 0

    # terminate training if it gets this accurate cuz it might be overfitting
    termination_acc = 1
    loss_history = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.gc1 = GraphConvolution(500, self.hidden_size)
        # self.gc2 = GraphConvolution(self.hidden_size, self.hidden_size)
        self.gc3 = GraphConvolution(self.hidden_size, 3)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, X):
        X = F.relu(self.gc1(X, self.adj))
        X = F.dropout(X, self.dropout, training=self.training)
        # X = F.relu(self.gc2(X, self.adj))
        # X = F.dropout(X, self.dropout, training=self.training)
        X = self.gc3(X, self.adj)
        return F.log_softmax(X, dim=1)

    def load_adj(self, adj):
        self.adj = adj.to(self.device)

    def train(self, X, y, idx_train):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        self.training = True

        accuracy_evaluator = Evaluate_Accuracy("training evaluator", "")

        inputs = torch.tensor(np.array(X), device=self.device, dtype=torch.float32)
        y_true = torch.tensor(np.array(y), device=self.device, dtype=torch.long)

        for epoch in range(self.max_epoch):
            optimizer.zero_grad()
            y_pred = self.forward(inputs)
            train_loss = loss_function(y_pred[idx_train], y_true[idx_train])
            train_loss.backward()
            optimizer.step()

            self.loss_history.append(train_loss.item())
            if epoch % 5 == 0 or epoch == self.max_epoch - 1:
                accuracy_evaluator.data = {
                    "true_y": y_true.cpu(),
                    "pred_y": y_pred.cpu().max(1)[1],
                }
                accuracy = accuracy_evaluator.evaluate()
                print(
                    "Epoch:", epoch, "Accuracy:", accuracy, "Loss:", train_loss.item()
                )

            if accuracy > self.termination_acc:
                break

    def test(self, X):
        self.training = False
        inputs = torch.tensor(np.array(X), device=self.device, dtype=torch.float32)
        outputs = self.forward(inputs).max(1)[1]
        return outputs

    def run(self):
        start = time.perf_counter()
        print("method running...")

        idx_train = self.data["train_test"]["idx_train"]
        idx_test = self.data["train_test"]["idx_test"]
        self.load_adj(self.data["graph"]["utility"]["A"])

        print("--start training...")

        self.train(self.data["graph"]["X"], self.data["graph"]["y"], idx_train)

        print("--start testing...")
        pred_y = self.test(self.data["graph"]["X"])
        end = time.perf_counter()
        print((end - start) / 60, " minutes elapsed")
        return (
            {
                "pred_y": pred_y.cpu()[idx_train],
                "true_y": self.data["graph"]["y"][idx_train],
            },
            {
                "pred_y": pred_y.cpu()[idx_test],
                "true_y": self.data["graph"]["y"][idx_test],
            },
            self.loss_history,
        )

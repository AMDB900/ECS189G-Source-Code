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
from torch_geometric.nn import GCNConv

class Method_GCN(method, nn.Module):
    data = None
    glove_embeddings = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_epoch = 150
    learning_rate = 1e-3

    max_review_length = 120
    hidden_size = 35

    # terminate training if it gets this accurate cuz it might be overfitting
    termination_acc = 0.88
    loss_history = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.conv1 = GCNConv(10, self.hidden_size)
        self.conv2 = GCNConv(self.hidden_size, 2)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, X):
        x, edge_index = X.x, X.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy("training evaluator", "")

        inputs = torch.tensor(np.array(X), device=self.device, dtype=torch.float32)
        y_true = torch.tensor(np.array(y), device=self.device, dtype=torch.long)

        for epoch in range(self.max_epoch):
            optimizer.zero_grad()
            y_pred = self.forward(inputs)
            train_loss = loss_function(y_pred, y_true)
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
                    "Epoch:",
                    epoch,
                    "Accuracy:",
                    accuracy,
                    "Loss:",
                )

            if accuracy > self.termination_acc:
                break

    def test(self, X):
        outputs = self.forward(X)
        return outputs

    def run(self):
        start = time.perf_counter()
        print("method running...")
        print("--start training...")
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

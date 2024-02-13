"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Method_ORL(method, nn.Module):
    data = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # it defines the max rounds to train the model
    max_epoch = 40
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    loss_history = []

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.conv1 = nn.Conv2d(1, 24, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 24, 5)
        self.fc1 = nn.Linear(12000, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 40)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy("training evaluator", "")

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(
            self.max_epoch
        ):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(
                self.input_tensor(X)
            )
            # convert y to torch.tensor as well
            y_true = self.out_tensor(y)
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            self.loss_history.append(train_loss.item())

            if epoch % 5 == 0:
                accuracy_evaluator.data = {
                    "true_y": y_true.cpu(),
                    "pred_y": y_pred.cpu().max(1)[1],
                }
                accuracy = accuracy_evaluator.evaluate()
                print(
                    "Epoch:", epoch, "Accuracy:", accuracy, "Loss:", train_loss.item()
                )

    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(
            torch.tensor(self.input_tensor(X))
        )
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def input_tensor(self, X):
        input_tensor = torch.tensor(np.array(X), device=self.device, dtype=torch.float32)
        input_tensor = input_tensor.permute(0, 3, 1, 2)
        return input_tensor[:, 0:1, :, :]
    
    def out_tensor(self, y):
        indexed = [i - 1 for i in y]
        return torch.tensor(np.array(indexed), device=self.device, dtype=torch.long)

    def run(self):
        print("method running...")
        print("--start training...")
        self.train(self.data["train"]["image"], self.data["train"]["label"])
        print("--start testing...")
        pred_y_train = self.test(self.data["train"]["image"])
        pred_y_test = self.test(self.data["test"]["image"])
        return (
            {
                "pred_y": pred_y_train.cpu(),
                "true_y": [i - 1 for i in self.data["train"]["label"]],
            },
            {
                "pred_y": pred_y_test.cpu(),
                "true_y": [i - 1 for i in self.data["test"]["label"]],
            },
            self.loss_history,
        )

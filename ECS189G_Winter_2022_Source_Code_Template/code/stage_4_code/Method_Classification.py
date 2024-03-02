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

class Method_Classification(method, nn.Module):
    data = None
    glove_embeddings = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # it defines the max rounds to train the model
    max_epoch = 5
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-5
    hidden_size = 100
    batch_size = 200
    input_size = 50
    num_classes = 2
    num_layers = 1
    loss_history = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        self.glove_embeddings = self.load_glove("data/stage_4_data/glove.6B.50d.txt")

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, X):
        batch_size = X.shape[1]
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(X, hidden)
        out = self.fc(out[:, -1, :])
        return out
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device).requires_grad_()
        return hidden
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
        max_len = max(len(seq) for seq in X)
        X_padded = []
        for seq in X:
            padded_seq = []
            for token in seq:
                if token in self.glove_embeddings:
                    padded_seq.append(self.glove_embeddings[token])
                else:
                    random_value = np.random.rand(50)
                    self.glove_embeddings[token] = random_value
                    padded_seq.append(random_value)
            X_padded.append(padded_seq + [np.zeros(50)] * (max_len - len(padded_seq)))
        return np.array(X_padded)

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy("training evaluator", "")
        dataset = TensorDataset(
            torch.tensor(self.data_preprocess(X), device=self.device, dtype=torch.float32),
            torch.tensor(np.array(y), device=self.device, dtype=torch.long),
        )
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself

        for epoch in range(self.max_epoch):  # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            all_y_true = []
            all_y_pred = []
            with torch.set_grad_enabled(True):
                for x_tensor, y_true in train_loader:
                    y_pred = self.forward(x_tensor)
                    # print(y_true.shape)
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
                    all_y_pred.extend(y_pred.cpu().detach())
                    all_y_true.extend(y_true.cpu().detach())
                    self.loss_history.append(train_loss.item())
            all_y_pred_tensor = torch.stack(all_y_pred)
            accuracy_evaluator.data = {
                "true_y": all_y_true,
                "pred_y": all_y_pred_tensor.max(1)[1],
            }
            accuracy = accuracy_evaluator.evaluate()
            print(
                "Epoch:", epoch, "Accuracy:", accuracy, "Loss:", train_loss.item()
            )

    def test(self, X):
        test_loader = DataLoader(torch.tensor(self.data_preprocess(X), device=self.device, dtype=torch.float32), batch_size=250)

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

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

class Method_Generation(method, nn.Module):
    data = None
    glove_embeddings = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    hidden_size = 25
    batch_size = 500
    input_size = num_classes = 5264
    num_layers = 1
    loss_history = []
    end_token = "ENDCHAR"

    vocabulary = {}

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.rnn = nn.RNN(
            self.input_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def make_vocabulary(self, X):
        vocabulary = set(word for window in X for word in window)
        vocabulary.add('ENDTOKEN')
        return {word: i for i, word in enumerate(vocabulary)}

    def to_one_hot(self, X):
        num_words = len(self.vocabulary)
        input_length = len(X[0])
        one_hot_tensor = np.zeros((len(X), input_length, num_words))

        for i, window in enumerate(X):
            for j, word in enumerate(window):
                word_index = self.vocabulary.get(word)
                if word_index == None:
                    continue
                one_hot_tensor[i, j, word_index] = 1

        return one_hot_tensor
    
    # takes the output of the foward() function of the model
    def to_word(self, tensor):
        word_id = tensor.max(1)[1][0]
        return list(self.vocabulary.keys())[word_id.item()]

    def forward(self, X):
        out, _ = self.rnn(X)
        out = self.fc(out[:, -1, :])
        return out

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy("training evaluator", "")

        self.vocabulary = self.make_vocabulary(X)

        dataset = TensorDataset(
            torch.tensor(self.to_one_hot(X), device=self.device, dtype=torch.float32),
            torch.tensor(np.array([self.vocabulary[word[0]] for word in y]), device=self.device, dtype=torch.long),
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
                    running_loss / len(train_loader),
                )

    # the input is the first n words of the joke, the output should be the whole joke generated
    # basically have the model predict words until it reaches the ENDTOKEN token
    def test(self, X):
        window_tensors = torch.tensor(self.to_one_hot(X), device=self.device, dtype=torch.float32)
        for i, window_tensor in enumerate(window_tensors):
            if i == 5: break
            current_token = ""
            output = " ".join(X[i])
            loop = 0
            while True:
                if loop > 1000:
                    print("took too long")
                    break
                out_tensor = self.forward(window_tensor.unsqueeze(0))
                current_token = self.to_word(out_tensor)
                if current_token == "ENDTOKEN": break
                output += " " + current_token
                window_tensor = torch.cat((window_tensor[1:], out_tensor), dim=0)
                loop += 1
            print(output)


    def run(self):
        start = time.perf_counter()
        print("method running...")
        print("--start training...")
        self.train(self.data["train"]["X"], self.data["train"]["y"])
        print("--start testing...")
        self.test(self.data["test"]["X"])
        end = time.perf_counter()
        print((end - start)/60, " minutes elapsed")
        return (
            self,
            self.loss_history,
        )

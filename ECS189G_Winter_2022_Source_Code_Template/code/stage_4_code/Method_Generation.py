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
import pickle

class Method_Generation(method, nn.Module):
    data = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 5000
    learning_rate = 1e-3
    max_epoch = 300

    hidden_size = 200
    num_layers = 1

    input_size = num_classes = 5264
    loss_history = []
    vocabulary = {}

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        with open("data/stage_4_data/text_generation/vocabulary.pkl", "rb") as f:
            self.vocabulary = pickle.load(f)

        self.rnn = nn.RNN(
            self.input_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, X):
        h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        out, _ = self.rnn(X, h0)
        out = self.fc(out[:, -1, :])
        return out

    # the vocabulary is a dict of words to indexes to be used in one hot encoding
    # def make_vocabulary(self, X):
    #     word_set = set(word for window in X for word in window)
    #     word_set.add("ENDTOKEN")
    #     vocabulary = {word: i for i, word in enumerate(word_set)}
    #     with open("data/stage_4_data/text_generation/vocabulary.pkl", "wb") as f:
    #         pickle.dump(vocabulary, f)
    #     return vocabulary

    # input: a list of lists of words
    # output: a 3d tensor of shape (number of samples, number of words, embedding size)
    def to_one_hot(self, X: list[list[str]]):
        num_words = len(self.vocabulary)
        input_length = len(X[0])
        one_hot_tensor = np.zeros((len(X), input_length, num_words))

        for i, window in enumerate(X):
            for j, word in enumerate(window):
                word_index = self.vocabulary.get(word)
                if word_index == None:
                    continue
                one_hot_tensor[i, j, word_index] = 1

        return torch.tensor(one_hot_tensor, device=self.device, dtype=torch.float32)

    # input: the output of the foward() function of the model
    # output: the string word that the model predicted
    def to_word(self, tensor) -> str:
        word_id = tensor.max(1)[1][0]
        return list(self.vocabulary.keys())[word_id.item()]

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy("training evaluator", "")

        # self.vocabulary = self.make_vocabulary(X)
        dataset = TensorDataset(
            self.to_one_hot(X),
            torch.tensor(
                np.array([self.vocabulary[words[0]] for words in y]),
                device=self.device,
                dtype=torch.long,
            ),
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
    def generate(self, words: list[str]) -> str:
        window_tensor = self.to_one_hot([words])
        current_token = ""
        output = " ".join(words)
        loop = 0
        while True:
            loop += 1
            if loop > 50:
                output += " [TERMINATED]"
                break

            model_output = self.forward(window_tensor)
            current_token = self.to_word(model_output)
            next_tensor = self.to_one_hot([[current_token]])

            if current_token == "ENDTOKEN":
                break

            window_tensor = torch.cat((window_tensor[:, 1:, :], next_tensor), dim=1)
            output += " " + current_token
        return output

    def test(self, X: list[list[str]]) -> list[str]:
        generations = []
        for starter in X:
            output = self.generate(starter)
            generations.append(output)
        return generations

    def run(self):
        start = time.perf_counter()
        print("method running...")
        print("--start training...")
        self.train(self.data["train"]["X"], self.data["train"]["y"])
        print("--start testing...")
        pred_y_test = self.test(self.data["test"]["X"])
        end = time.perf_counter()
        print((end - start)/60, " minutes elapsed")
        return (
            pred_y_test,
            self.loss_history,
        )

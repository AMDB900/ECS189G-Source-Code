from code.base_class.method import method
import torch
import torch.nn as nn
from code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchtext.vocab import Vectors


class Method_RNN(method, nn.Module):
    data = None
    max_epoch = 5
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    batch_size = 12500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(Method_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, text, h0):
        output, hn = self.rnn(text, h0)
        output = self.fc(output.view(-1, self.hidden_size))
        return output, hn

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy()
        dataset = TensorDataset(self.x_tensor(X), self.y_tensor(y))
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.max_epoch):
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, labels = batch.data, batch.labels
                pred = self(inputs)
                train_loss = loss_function(pred, self.labels)
                train_loss.backward()
                optimizer.step()

                self.loss_history.append(train_loss.item())

                if epoch % 5 == 0 or epoch == self.max_epoch - 1:
                    accuracy = self.evaluate_accuracy(inputs, labels)
                    print("Epoch:", epoch, "Accuracy:", accuracy, "Loss:", train_loss.item())

    def test(self, test_data):
        y_pred_list = []
        for inputs in test_data:
            outputs = self(inputs)
            y_pred_list.append(outputs.max(1)[1])
        return torch.cat(y_pred_list)

    def X_tensor(self, X):
        return torch.tensor(np.array(X), device=self.device, dtype=torch.long)

    def y_tensor(self, y):
        return torch.tensor(np.array(y), device=self.device, dtype=torch.long)

    def run(self):
        print("method running...")
        print("--start training...")
        self.train(self.data["train"]["text"], self.data["train"]["label"])
        print("--start testing...")
        pred_y_train = self.test(self.data["train"]["text"])
        pred_y_test = self.test(self.data["test"]["text"])
        return (
            {
                "pred_y": pred_y_train,
                "true_y": self.data["train"]["label"],
            },
            {
                "pred_y": pred_y_test,
                "true_y": self.data["test"]["label"],
            },
            self.loss_history,
        )

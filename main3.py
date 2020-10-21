import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn


class Solver(nn.Module):

    def __init__(self):
        super().__init__()

        self.liner_1 = nn.Linear(1, 100)
        self.liner_2 = nn.Linear(100, 1)
        self.tanh = nn.Tanh()
        self.optimazer = torch.optim.SGD(self.parameters(), lr=0.01)
        self.learning_step_n = 1000

    def forward(self, input):
        hidden = self.liner_1(input)
        hidden = self.tanh(hidden)
        output = self.liner_2(hidden)
        return output

    def learning(self, x_data, y_data):
        for _ in range(self.learning_step_n):
            loss = torch.mean((self.forward(x_data - y_data)) ** 2)
            loss.backward()
            self.optimazer.step()
            self.optimazer.zero_grad()


def main():
    x_data = torch.linspace(-5, 5, steps=300)
    nu, sigma = torch.tensor(0.2), torch.tensor(0.3)
    noise = torch.tensor([torch.normal(nu, sigma) for _ in range(300)])
    y_data = torch.sin(x_data) + noise
    x_data = x_data.reshape(300, 1)
    y_data = y_data.reshape(300, 1)

    solver = Solver()
    solver.learning(x_data, y_data)

    plt.scatter(x_data.numpy(), y_data.numpy())
    plt.plot(x_data.numpy(), solver(x_data).detach().numpy(), 'r')
    plt.show()


if __name__ == '__main__':
    main()

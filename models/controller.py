""" Define controller """
import torch
import torch.nn as nn

class Controller(nn.Module):
    """ Controller """
    def __init__(self, sampledhidden, hidden, actions):
        super().__init__()
        self.fc = nn.Linear(sampledhidden + hidden, actions)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        print(cat_in.shape)
        return self.fc(cat_in)

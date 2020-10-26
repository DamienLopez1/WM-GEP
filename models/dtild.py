""" Define controller """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Dtild(nn.Module):
    """ Controller """
    def __init__(self, hidden, actions,output):
        super().__init__()
        self.fc = nn.Linear(hidden + actions, output)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        return F.relu(self.fc(cat_in))

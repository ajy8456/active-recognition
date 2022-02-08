import torch
import torch.nn as nn
import torch.nn.functional as F
from googlenet import *

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(256, 35, bias=True)

    def forward(self, a, p, num_samples):
        input = torch.cat((a, p), dim=1)                #Check
        output = self.layer1(input)
        output = torch.clamp(output, 0, 1)              #clamp
        output = F.normalize(output, dim=0)             #L1-normalize
        output = torch.multinomial(output, num_samples)
        return output

class Sensor(nn.Module):
    def __init__(self):
        super(Sensor, self).__init__()
        self.layer1 = nn.Linear(2, 16, bias=True)
        self.CNNfeat = GoogleNet()
        self.layer2 = nn.Linear(1040, 256, bias=True)
        self.layer3 = nn.Linear(256, 256, bias=True)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout()                    #put parameter 0?
        self.dropout2 = nn.Dropout()                    #put parameter 0?

    def forward(self, m, v):
        m = self.relu(self.layer1(m))
        v = self.dropout1(self.CNNfeat(v))
        input = torch.cat((m,v), dim=1)

        output = self.relu(self.layer2(input))
        output = self.batch_norm(self.layer3(output))
        output = self.dropout2(output)
        return output

class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()
        self.relu = nn.ReLu()
        self.layer1 = nn.Linear(256, 256)

    def forward(self, m, v):
        # Can't understand what + means, so I just put concatenate
        # Plus we have to check whether append means stack or cat
        input = torch.cat((m,v), dim=1)
        output = self.relu(input)
        output = self.layer1(output)
        return output

class LookAhead(nn.Module):
    def __init__(self):
        super(LookAhead, self).__init__()
        self.layer1 = nn.Linear(260, 100)
        self.layer2 = nn.Linear(100, 256)
        self.relu = nn.ReLU()

    def forward(self, a, m, p):
        # Same we have to check whether append means stack or cat
        input = torch.cat((a, m), dim=1)
        input = torch.cat((input, p), dim=1)
        output = self.relu(self.layer1(input))
        output = self.layer2(output)
        return output


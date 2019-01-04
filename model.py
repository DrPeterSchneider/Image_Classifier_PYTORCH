import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models


def model(architecture, processor, learning_rate,  hidden_units):
    if architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(1024, hidden_units[0])),
                           ('relu1', nn.ReLU()),
                           ('dopout1', nn.Dropout(p=0.4)),
                           ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                           ('relu2', nn.ReLU()),
                           ('dropout2', nn.Dropout(p=0.4)),
                           ('fc3', nn.Linear(hidden_units[1],102)),
                           ('output', nn.LogSoftmax(dim=1))
                           ]))
        model.classifier=classifier

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        if processor == 'GPU':
            model.cuda()
        elif processor  == 'CPU':
            model

    elif architecture == 'densenet169':
        model = models.densenet169(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(1024, hidden_units[0])),
                           ('relu1', nn.ReLU()),
                           ('dopout1', nn.Dropout(p=0.4)),
                           ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                           ('relu2', nn.ReLU()),
                           ('dropout2', nn.Dropout(p=0.4)),
                           ('fc3', nn.Linear(hidden_units[1],102)),
                           ('output', nn.LogSoftmax(dim=1))
                           ]))
        model.classifier=classifier


        if processor == 'GPU':
            model.cuda()
        elif processor  == 'CPU':
            model
    
    return model
    
#Test with the commands below
#print(model('densenet169', 'CPU', 0.001, [512, 256]))
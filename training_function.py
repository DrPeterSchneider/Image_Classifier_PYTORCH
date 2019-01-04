import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from model import model


def training_function1(data_dir, model, processor, epochs, learning_rate, architecture, hidden_units):
 
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(25),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    trainloader, validloader, testloader

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate) 
    steps=0
    running_loss=0
    print_every=40
        
  
    for e in range(epochs):
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            inputs, labels = Variable(inputs), Variable(labels)
            steps += 1

            optimizer.zero_grad()


            # Move input and label tensors to the GPU
            if processor == 'GPU':
                inputs, labels = inputs.cuda(), labels.cuda()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            if steps % print_every == 0:
            # Model in inference mode, dropout is off
                model.eval()

                accuracy = 0
                test_loss = 0
                for ii, (inputs, labels) in enumerate(validloader):
                        
                    # Move input and label tensors to the GPU
                    if processor == 'GPU':
                        inputs, labels = inputs.cuda(), labels.cuda()
                        
                    inputs, labels = Variable(inputs), Variable(labels)


                    output = model.forward(inputs)
                    test_loss += criterion(output, labels).data[0]

                    # Calculating the accuracy
                    # Model's output is log-softmax, take exponential to get the probabilities
                    ps = torch.exp(output).data
                    # Class with highest probability is our predicted class, compare with true label
                    equality = (labels.data == ps.max(1)[1])
                    # Accuracy is number of correct predictions divided by all predictions, just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

            running_loss = 0

        # Make sure dropout is on for training
            model.train()
  

    checkpoint_information = {'architecture': architecture,
                              'processor': processor,
                              'learning_rate': learning_rate,
                              'hidden_units': hidden_units,
                              'state_dict': model.state_dict()}

    torch.save(checkpoint_information, 'checkpoint.pth')
import numpy as np
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from model import model
from training_function import training_function1 
import argparse 

print("Welcome to the Image Classifier!")
print("Your command should look like this: python train.py 'flowers' 'densenet121' 'GPU' 0.001 -c 512 -c 256 3")

parser = argparse.ArgumentParser()
parser.add_argument("image_dataset", help = "dir of dataset of images divided in train, validation and test images")
parser.add_argument("pretrained_arch", help = "choose between 'densenet121' and 'densenet169', mind ''")
parser.add_argument("processor", help = "Choose between 'CPU' and 'GPU', mind ''")
parser.add_argument("learning_rate", help = "Learningrate for network training", type=float)
parser.add_argument('-c', action='append', help="List of 2 IntNumbers -l x -l y  which represent the hidden units. Mind Input=1024, Output=102, 3 Layers", type=int)
parser.add_argument("epochs", help= "Number of training Epochs", type=int)
args = parser.parse_args()

print(args.pretrained_arch, args.processor, args.learning_rate, args.c )
training_function1(args.image_dataset, model(args.pretrained_arch, args.processor, args.learning_rate, args.c ), args.processor, args.epochs, args.learning_rate, args.pretrained_arch, args.c)

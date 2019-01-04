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
from predict_function import process_image
from predict_function import predict
from predict_function import print_topk
import argparse 

print("\n", "Welcome to the Image Classifier!")
print("Your command should look like this: python predict.py 'cat_to_name.json' 'checkpoint.pth' flowers/test/10/image_07090.jpg 5 'GPU'", "\n")


parser = argparse.ArgumentParser()
parser.add_argument("json_categories", help="JSON file with dir(if different dir) to name classes, mind ''")
parser.add_argument("model_checkpoint", help="PTH file with dir(if different dir) to load saved checkpoint, mind ''")
parser.add_argument("test_image", help="image file with dir(if different dir) which should be classified")
parser.add_argument("topk", help="IntNumber k<102 for Top k classification by probability", type=int)
parser.add_argument("processor", help= "Choose between 'CPU' and 'GPU', mind ''")
args = parser.parse_args()


if args.processor=='CPU':
    checkpoint_information = torch.load(args.model_checkpoint, map_location='cpu')
else:
    checkpoint_information = torch.load(args.model_checkpoint)
architecture = checkpoint_information['architecture']
learning_rate = checkpoint_information['learning_rate']
hidden_units = checkpoint_information['hidden_units']  


result=print_topk(args.json_categories, args.model_checkpoint,(args.test_image), model(architecture, args.processor, learning_rate, hidden_units), args.topk, args.processor)
print("The probability for {} is {}.".format(result[1][0], result[0][0]), "\n", "The TopK classes and probabilities are:", "\n")
i=0
while i <(args.topk):
    print("The probability for {} is {}.".format(result[1][i], result[0][i]), "\n")
    i=i+1

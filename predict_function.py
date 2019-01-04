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



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    from PIL import Image
    mean = [0.485, 0.456, 0.406]
    stdv = [0.229, 0.224, 0.225]
    img = Image.open(image)
    if img.size[0]>=img.size[1]:
        img.thumbnail((10000,256))
    else:
        img.thumbnail((256,10000))

    half_the_width = img.size[0] / 2
    half_the_height = img.size[1] / 2
    img = img.crop(
        (
            half_the_width - 112,
            half_the_height - 112,
            half_the_width + 112,
            half_the_height + 112
        )
    )

    np_image = np.array(img)
    img = np_image/255
    img=(img-mean)/stdv

    img=img.transpose((2,0,1))
    return img

def predict(json_map, checkpoint,image_path, model, topk, processor): 
#''' Predict the class (or classes) of an image using a trained deep learning model.'''
# Test out your network!
                      
    if processor=='CPU':
        checkpoint_information = torch.load(checkpoint, map_location='cpu')
    else:
        checkpoint_information = torch.load(checkpoint)
    
    
    model.load_state_dict(checkpoint_information['state_dict'])
    model.eval()
    a = process_image(image_path)
    y = np.expand_dims(a, axis=0)
    if processor=='CPU':
        img = torch.from_numpy(y)
    else:
        img = torch.from_numpy(y).cuda()
    output = model.double()(Variable(img, volatile=True))
    ps = torch.exp(output)
    ps_top5 = torch.topk(ps,topk)
    probs = ps_top5[0]
    classes = ps_top5[1]
    
    import json
    with open(json_map, 'r') as f:
        cat_to_name = json.load(f)
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    train_transforms = transforms.Compose([transforms.RandomRotation(25),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)    
    xx=train_data.class_to_idx

    class_to_idx_dict=train_data.class_to_idx
    key_value_exchange_dict = dict((v,k) for k,v in xx.items())
        
    probabilities =  probs.data.cpu().numpy().tolist()[0]
    plant_classes = classes.data.cpu().numpy().tolist()[0]
    for i in range(len(probabilities)):
        plant_classes[i]=key_value_exchange_dict[plant_classes[i]]
    
    
    return probabilities, plant_classes, cat_to_name

def print_topk(json_map, checkpoint, image_path ,model, topk, processor):
    
    probabilities, plant_classes, cat_to_name = predict(json_map, checkpoint,image_path, model, topk, processor)
        
    indexlist=[]
    for i in range(len(probabilities)):
        indexlist.append(cat_to_name['{}'.format(plant_classes[i])])
    
    result_list=(indexlist)
    topk_probs=(predict(json_map, checkpoint, image_path, model, topk, processor)[0])
    return topk_probs, result_list
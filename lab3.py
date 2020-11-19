import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable
from torchvision.models.alexnet import alexnet
from torch.utils.data.sampler import SubsetRandomSampler
import csv
import time
from torch2trt import torch2trt

images=[]

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

#data_dir = '~/lab3/jetson-inference/data'
img_path = './data/7.jpg'
image = Image.open(img_path)
images.append(image)

#Transforms image for model input
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Model loading
model = alexnet(pretrained=True).eval().cuda()
model_trt = torch2trt(model, [x])

#Read classes because our model is pretrained
classes=[]
with open('imagenet.txt', 'r') as fd:
    reader = csv.reader(fd)
    for row in reader:
        classes.append(row)

#Transforming image using test_transforms and predicting class using model
def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    timest = time.time()
    output = model_trt(input)
    print("processing {}".format(time.time()-timest))
    index = output.data.cpu().numpy().argmax()
    return index

#process our images
def processing(images):
    fig=plt.figure(figsize=(10,10))
    
    for ii in range(len(images)):
        sub = fig.add_subplot(1, len(images), ii+1)
        index = predict_image(images[ii])
        sub.set_title("class " + str(classes[index]))
        plt.axis('off')
        plt.imshow(image)
        plt.savefig(str(index)+'.png')
        #plt.show()

processing(images)

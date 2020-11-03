import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
import time

# create some regular pytorch model...
timest = time.time()
model = alexnet(pretrained=True).eval().cuda()
print("load time {}".format(time.time()-timest))

# create example data
x = torch.ones((1, 3, 224, 224)).cuda()

#timest = time.time()
#y = model(x)
#print(time.time()-timest)


#timest_trt = time.time()
# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])

torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
#y_trt = model_trt(x)
#print(time.time()-timest_trt)
# check the output against PyTorch
#print(torch.max(torch.abs(y - y_trt)))

#print("y {}".format(y))

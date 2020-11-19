import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
from torchvision.models.alexnet import alexnet
import time
from PIL import Image
from torchvision import transforms

#load image from folder
img_path="7.jpg"
image = Image.open(img_path)
image = transforms.ToTensor()(image).unsqueeze_(0).cuda()

# create some regular pytorch model...
timest = time.time()
model = alexnet(pretrained=True).eval().cuda()
print("load time {}".format(time.time()-timest))

# create example data
#x = torch.ones((1, 3, 224, 224)).cuda()

timest = time.time()
y = model(image)

print(time.time()-timest)

# convert to TensorRT feeding sample data as input
#model_trt = torch2trt(model, [image])
#y_trt = model_trt(x)
#timest_trt = time.time()
#torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
#torch.save(model.state_dict(), 'alexnet.pth')

#print(time.time()-timest_trt)
# check the output against PyTorch
#print(torch.max(torch.abs(y - y_trt)))

#print("y {}".format(y))

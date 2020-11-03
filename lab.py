import jetson.inference
import jetson.utils
from torchvision.models.alexnet import alexnet

import argparse

# parse the command line
parser = argparse.ArgumentParser()
#parser.add_argument("filename", type=str, help="filename of the image to process")
#parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect.")

parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
args = parser.parse_args()

# load an image (into shared CPU/GPU memory)
#img = jetson.utils.loadImage(args.filename)

input = jetson.utils.videoSource("csi://0")
output = jetson.utils.videoOutput("display://0")
font = jetson.utils.cudaFont()

# load the recognition network for imagenet
#net = jetson.inference.imageNet(args.network)

# for detectNet 
net = jetson.inference.detectNet(args.network)
opt="box,labels,conf"

# classify the image
#class_idx, confidence = net.Classify(img)
#class_desc = net.GetClassDesc(class_idx)
#print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))

#live-streaming
while True:
    img = input.Capture()
    
	# overlay the result on the image - first mode (imagenet)
    #class_id, confidence = net.Classify(img)
    #class_desc = net.GetClassDesc(class_id)
    #font.OverlayText(img, img.width, img.height, "{:05.2f}% {:s}".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)
	
    # detect objects in the image (with overlay) - second mode
    detections = net.Detect(img, overlay=opt)
    # print the detections
    print("detected {:d} objects in image".format(len(detections)))
    for detection in detections:
        print(detection)

	# render the image
    output.Render(img)

	# update the title bar - imagenet
    #output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # update the title bar - detectnet
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

	# print out performance info
    net.PrintProfilerTimes()

	# exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break



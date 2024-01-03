import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import copy

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
torch.set_default_device(device)

imsize = 512 if torch.cuda.is_available() or torch.backends.mps.is_available() else 128  

loader = transforms.Compose([
    transforms.Resize(imsize),  
    transforms.ToTensor(),
    ])  

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
    
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  
    image = image.squeeze(0)      # remove the fake batch dimension
    image = transforms.ToPILImage(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='The configs')
    parser.add_argument('--model',type=str, default='Gatys')
    args = parser.parse_args()
    
    style_img = image_loader("./style_images/wave_crop.jpg")
    content_img = image_loader("./input_images/dancing.jpg")
    
    if args.model == 'Gatys':
        output = gatys_style_transfer(net, net_normalization_mean, net_normalization_std, content_img, style_img, input_img)
    elif args.model == 'LapStyle':
        output = lap_style_transfer()
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
from torch import optim
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import vgg19, VGG19_Weights

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def deprocess_img(img):
    img *= torch.tensor(mean).view(-1, 1, 1)
    img += torch.tensor(std).view(-1, 1, 1)
    return img
    
img = Image.open('./input_images/dancing.jpg')
img_style = Image.open('./style_images/picasso.jpg')


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std),
])

img = transform(img).to(device)
img_style = transform(img_style).to(device)
img = img.unsqueeze(0)
img_style = img_style.unsqueeze(0)

vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT).features.to(device)

content_layer = [29]
style_layers = [1, 6, 11, 18, 25]
output_layers = content_layer + style_layers    

def gram_matrix(input):
    a, b, c, d = input.size()  
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d) 

    G = torch.mm(features, features.t())  
    return G.div(a * b * c * d)
    

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16)
        self.features = nn.ModuleList(features).eval() 

    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in output_layers:
                results.append(x)
        return results      
    
model = Vgg16()

# get style and content targets
content_t = model(img)[-1]
style_ = model(img_style)[:-1]
style_t = []
for st in style_:
    style_t.append(gram_matrix(st))
    
criterion = nn.MSELoss()

class LapStyle(torch.nn.Module):
    def __init__(self):
        super(LapStyle, self).__init__()
        self.avg_pool = nn.AvgPool2d(3, stride=3)
        self.laplacian_filter = nn.Conv2d(3, 3, 3, 1, bias=False)
        laplacian = torch.tensor([[[0.,-1.,0.],
                                  [-1.,4.,-1.],
                                  [0.,-1.,0.]],
                                  [[0.,-1.,0.],
                                  [-1.,4.,-1.],
                                  [0.,-1.,0.]],
                                  [[0.,-1.,0.],
                                  [-1.,4.,-1.],
                                  [0.,-1.,0.]],
                                  ])
        laplacian = laplacian.unsqueeze(0)
        self.laplacian_filter.weight = nn.Parameter(laplacian)
        self.laplacian_filter.weight.requires_grad = False

        features = list(vgg16)
        self.features = nn.ModuleList(features).eval() 

    def forward(self, input):
        results = []

        x = input.clone()
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in output_layers:
                results.append(x)
        
        x = input.clone()
        x = self.avg_pool(x)
        x = self.laplacian_filter(x)
        results.append(x)
        return results

lap_out_image = img.clone().detach().requires_grad_(True)
# optimizer = optim.Adam([lap_out_image], lr=0.1)
optimizer = optim.LBFGS([lap_out_image])
model = LapStyle().to(device)

lap_target = model(lap_out_image)[-1]

model.eval()

num_steps = 20
content_weight, style_weight, lap_weight = 1e-20, 1e+20, 1e+20

epoch = [0]
while epoch[0] <= num_steps:

    def closure():
        
        with torch.no_grad():
                lap_out_image.clamp_(0, 1)
        
        optimizer.zero_grad()
        out = model(lap_out_image)

        content_loss = criterion(out[-2], content_t)

        style_loss = 0
        for style_layer, target_style in zip(out[:-2], style_t):
            style_loss += criterion(gram_matrix(style_layer), target_style)
        style_loss /= len(style_t)

        lap_loss = criterion(out[-1], lap_target)

        loss = style_loss * style_weight + content_loss * content_weight + lap_loss * lap_weight

        loss.backward(retain_graph=True)

        return style_loss * style_weight + content_loss * content_weight + lap_loss * lap_weight

    optimizer.step(closure)  


lap_out_image = deprocess_img(lap_out_image.detach().squeeze().cpu())
lap_out_image = lap_out_image.permute(1, 2, 0)

plt.title('Output image')
plt.imshow(torch.clip(lap_out_image, min=0, max=1))  
plt.show()
from torchvision import models
fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

from PIL import Image
import matplotlib.pyplot as plt
import torch

img = Image.open('./bird.png')
# plt.imshow(img); plt.show()

# Apply the transformations needed
# Preprocess it and normalize it!

#Resize image : 256 x 256
#CenterCrop it : (224 x 224)
#convert it to tentor
#normalize it with Imagenet specific values mean and std
import torchvision.transforms as T
trf = T.Compose([T.Resize(256),
                 T.CenterCrop(224),
                 T.ToTensor(), 
                 T.Normalize(mean = [0.485, 0.456, 0.406], 
                             std = [0.229, 0.224, 0.225])])
# What is T.Compose function?

#unsqueeze the image so that it becomes [1 x C x H x W] from [C x H x W]
# torch.unsqueeze() :  특정 위치 차원 추가
# torch.squeeze() : 특정 위치 차원 제거
# torch.view() : 3차원 텐서의 크기 변경
# view example : (2 x 2 x 3) tensor --> (? x 1 x 3)으로 변경?
# ft.view([-1, 1, 3])으로 변경 가능
inp = trf(img).unsqueeze(0)
#inp : input image that all preprocessed.

# Pass the input through the net
out = fcn(inp)['out']
print (out.shape)



import numpy as np
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
print (om.shape)
print (np.unique(om))

#om : output image (224 x 224)
#it is required to visualize it
#As a result, we use 'decode_segmap' function to visualize it

# Define the helper function
def decode_segmap(image, nc=21):
  
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  #astype : assign type
  
  #range(0, nc) : 0 ~ 21
  # l : 0 ~ 21  
  for l in range(0, nc):
    idx = image == l #What is this syntax 
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=2)
  return rgb


rgb = decode_segmap(om)
print(rgb.shape) # rgb : (224 x 224 x 3(rgv channel))

plt.imshow(rgb); 
plt.show()


def segment(net, path, show_orig=True, dev='cuda'):
  img = Image.open(path)
  if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
  # Comment the Resize and CenterCrop for better inference results
  trf = T.Compose([T.Resize(640), 
                   #T.CenterCrop(224), 
                   T.ToTensor(), 
                   T.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
  inp = trf(img).unsqueeze(0).to(dev)
  out = net.to(dev)(inp)['out']
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  rgb = decode_segmap(om)
  plt.imshow(rgb); plt.axis('off'); plt.show()

# segment(fcn, './horse.png')

dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
segment(dlab, './horse.png')
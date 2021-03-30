import torch
import torch.onnx as onnx
import torchvision.models as models

#Saving and Loading Model Weights.
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
# be sure to call model.eval() method before inferencing 
# to set the dropout and batch normalization layers to evaluation mode. 
# Failing to do this will yield inconsistent inference results.


#Load model
# model.load_state

# model = torch.load('model_weights.pth')

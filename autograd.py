#When training neural networks, the most frequently used algorithm is 
#back propagation. 
#In this algorhim, parameters are adjusted according to the gradient
#of the loss function with respect to the given parameter.

import torch

x = torch.ones(5) #input tensor
y = torch.zeros(3) #expected ouput(predicted output)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b #z : ground truth
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print('Gradient function for z =',z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

#compute gradients
#To optimize weights of parameters in the neural network, 
#we need to compute the derivatives of our loss function with 
#respect to parameters.
#To compute those derivatives, we call loss.backward()
loss.backward()
print(w.grad)
print(b.grad)

#Disabling gradient tracking
#In cases when we want to do only forward computation through the 
#network, we can stop tracking computations by no_grad() function.
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

#Same function can be utilzed by z.detach()
z = torch.matmul(x, w) + b
z_det = z.detach()
print(z.requires_grad)
print(z_det.requires_grad)

#When do we want to disable gradient tracking?
# 1. To mark some parameters in your neural network
# at frozen parameters. This is a very common scenario for
# finetuning a pretrained network.
# 
# 2. To speed up computations when you are only doing 
# forward pass, because computations on tensors that do not
# track gradients would be more efficient.

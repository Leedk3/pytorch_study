import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available else 'cpu'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input : 1 image channel
        # output : 6 ouput channels, 3x3 conv. kernel. 
        self.conv1 = nn.Conv2d(1, 6, 3)
        # input : 6 image channel
        # output : 16 ouput channels, 3x3 conv. kernel.         
        self.conv2 = nn.Conv2d(6, 16, 3)

        # an affine operation : y = wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # nn.functional.relu vs nn.ReLU()
        # nn.ReLU() creates an nn.Module which can add nn.Sequential model.
        # nn.functional.relu is just the functional API call.
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] #all dimensions except the batch dimention.
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net().to(device = device)
print(net)


# You just have to define forward function, and the backward
# function is automatically defined for you using autograd.
# you can use any of the Tensor operations in the forward function.
params = list(net.parameters())
print(len(params))
#See the parameters in the each layers.
for i in range(len(params)):
    print(f"{i},", params[i].size())

#RANDOM TEST
input = torch.randn(1, 1, 32, 32).to(device)
out = net(input)
print(out)
print("out size:", out.size())
#Zero the gradient buffers of all parameters 
#backprops with random gradients.
net.zero_grad()
out.backward(torch.randn(1, 10).to(device))

## NOTE
# torch.nn only supports mini-batches
# The entire torch.nn package only supports inputs that are 
# a mini-batch of samples and not a single sample.
# If you have a single sample, just use input.unsqueeze(0)
# to add a fake batch dimension.


# Loss Function.
output = net(input)
target = torch.randn(10).to(device)
target = target.view(1, -1)
loss = nn.MSELoss()(output, target)
print(loss)

#loss.grad_fn function represents from backward: 
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU


#Back propagation
# To backpropagate the error all we have to do is to loss.backward()
# You need to clear the existing gradients though, else
# gradients will be accumulated to existing gradients.

net.zero_grad()

print('conv1. bias grad before backward')
print(net.conv1.bias.grad)
loss.backward()
print('conv1. bias grad after backward')
print(net.conv1.bias.grad)


#Optimization
import torch.optim as optim

#create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

#in our training loop:
optimizer.zero_grad()
output = net(input)
loss = nn.MSELoss()(output, target)
loss.backward()
optimizer.step()
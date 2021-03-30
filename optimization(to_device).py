#Now that we have a model and data it's time to train, validate
#and test our model by optimizing it's parameters on our data.
#Training a model is an iterative process; in each iteration
#(called an epoch) the model makes a guess about the output,
#calculates the error in its guess, collects the derivatives of 
#the error with respect to its parameters, and optimizes these 
#parameters using gradient descent.
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device = device)

# Hyper-parameters
# Hyperparameters are adjustable parameters that let you control
# the model optimization process. Different hyperparameter values 
# can impact model training and convergence rates.
# Number of epochs : the number times to iterates over the dataset
# Batch size : the number of data sample seen by the model in each epoch
# Learning rate : how much to update models parameters at each batch/epoch
# Smaller values yield slow learning speed, while large values
# may result in unpredictable behavior during training.
learning_rate = 1e-3
batch_size = 64
epochs = 5

#Loss function
#Loss function measures the degree of dissimilarity of obtained
#result to the target value, and it is the loss function that
#we want to minimize during traning.
# i.e. nn.MSELoss(Mean Square Error), nnNLLLoss
loss_fn = nn.CrossEntropyLoss()

# optimizer
# optimiztion is the process of adjusting model parameters to reduce
# model error in each training step.
# optimization algorithm define how this process is performed.
# All optimization logic is encapsulated in the optimizer object
# Here, we use the SGD optimizer; additionally, there are many different
# optimizer available in PyTorch such as ADAM and RMSProp, 
# that work better for different kinds of models and data.
# i.e. ADAM, RMSProp, SGD optimizer..
# https://hyunw.kim/blog/2017/11/01/Optimization.html 
optimizer  = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Full implementation
# Call optimizer.zero_grad() to reset the gradients of model parameters.
# Gradients by default add up; to prevent double -counting, 
# we explicitly zero them at each iteration.
# Backpropagate the prediction loss with a call to loss.backward()
# Once we have our gradients, we call optimizer.step() to adjust
# the parameters by the gradients collected in the backward pass.

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        #compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") 

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Done!")


#Save and Load the model
import torch.onnx as onnx
import torchvision.models as models

torch.save(model.state_dict(), 'model_weights.pth')

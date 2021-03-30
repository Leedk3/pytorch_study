import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y : torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

#scatter , gather
#scatter(dim, index, source) --> Tensor
#scatter_(dim, intex, src) --> Tensor

# scatter example
x = torch.rand(2, 5)
print(x)

scatter_exp = torch.zeros(3,5).scatter_(0, torch.tensor([[0,1,2,0,0],[2,0,0,1,2]]), x)
print(scatter_exp)
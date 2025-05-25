import torch
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchvision.datasets import *
from torchvision.transforms import *
from typing import Union, Any, Tuple
import torchprofile

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet18, ResNet18_Weights

import numpy as np
import pickle

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.cnv3 = nn.Conv2d(6, 16, 5, stride=1)
        self.relu3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.cnv5 = nn.Conv2d(16, 120, 5, stride=1)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(120, 84)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(84, 10)

    def forward(self, image):
        self.out1 = self.relu1(self.cnv1(image))
        self.out2 = self.pool2(self.out1)
        self.out3 = self.relu3(self.cnv3(self.out2))
        self.out4 = self.pool4(self.out3)
        self.out5 = self.relu5(self.cnv5(self.out4))
        self.out5_flt = torch.flatten(self.out5, 1)
        self.out6 = self.relu6(self.fc6(self.out5_flt))
        self.out7 = self.fc7(self.out6)

        return self.out7


    def load_params(self, addr, device):
        wt = open(addr, 'rb')
        b = pickle.load(wt, encoding='latin1')
        wt.close()

        b0 = np.float32(b[0]['conv1.weights'])
        b1 = np.float32(b[0]['conv1.bias'].reshape(6))

        b2 = np.float32(b[3]['conv3.weights'])
        b3 = np.float32(b[3]['conv3.bias'].reshape(16))

        b4 = np.float32(b[6]['conv5.weights'])
        b5 = np.float32(b[6]['conv5.bias'].reshape(120))

        b6 = np.float32(b[9]['fc6.weights'].T)
        b7 = np.float32(b[9]['fc6.bias'].reshape(84))

        b8 = np.float32(b[11]['fc7.weights'].T)
        b9 = np.float32(b[11]['fc7.bias'].reshape(10))

        self.cnv1.weight = nn.Parameter(torch.from_numpy(b0).to(device))
        self.cnv1.bias = nn.Parameter(torch.from_numpy(b1).to(device))

        self.cnv3.weight = nn.Parameter(torch.from_numpy(b2).to(device))
        self.cnv3.bias = nn.Parameter(torch.from_numpy(b3).to(device))

        self.cnv5.weight = nn.Parameter(torch.from_numpy(b4).to(device))
        self.cnv5.bias = nn.Parameter(torch.from_numpy(b5).to(device))

        self.fc6.weight = nn.Parameter(torch.from_numpy(b6).to(device))
        self.fc6.bias = nn.Parameter(torch.from_numpy(b7).to(device))

        self.fc7.weight = nn.Parameter(torch.from_numpy(b8).to(device))
        self.fc7.bias = nn.Parameter(torch.from_numpy(b9).to(device))


def load_model(model_name: str, 
               dataset_name: str, 
               device: Union[torch.device, str] = torch.device('cuda'),
               ) -> nn.Module:
    
    if 'mnist' in dataset_name:
        model = LeNet().to(device)
        addr = r'./lenet5_mnist.pkl'
        model.load_params(addr, device)

    elif 'cifar' in dataset_name:
        full_name = dataset_name + "_" + model_name
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", full_name, pretrained=True)

    elif 'imagenet' in dataset_name:
        if model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)

        elif model_name == "resnet34":
            model = resnet34(weights=ResNet34_Weights.DEFAULT).to(device)
        
        elif model_name == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)

    return model.to(device)


def load_params(model: nn.Module, 
                addr: str, 
                device: Union[torch.device, str] = torch.device('cpu'),
                ) -> None:
    i = 0
    state_dict_load = torch.load(addr, map_location=device)
    sd = model.state_dict()
    for layer_name, _ in sd.items():
        if 'num_batches_tracked' not in layer_name:           
            sd[layer_name] = nn.Parameter(state_dict_load[list(state_dict_load)[i]].to(device))
        i += 1
    
    model.load_state_dict(sd)


def train(model: nn.Module,
          dataloader: DataLoader,
          criterion: nn.Module,
          optimizer: Optimizer,
          scheduler: LambdaLR,
          callbacks = None,
          device=torch.device('cuda')) -> None:
    
    model.train()

    for data in dataloader:
        inputs, targets = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        # Forward inference
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward propagation
        loss.backward()

        # Update optimizer and LR scheduler
        optimizer.step()
        scheduler.step()

        if callbacks is not None:
            for callback in callbacks:
                callback()


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device=torch.device('cuda')) -> float:
    
    model.to(device)
    model.eval()
    num_samples = 0
    num_correct = 0

    for data in dataloader:
        inputs, targets = data[0].to(device), data[1].to(device)

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

        #break

    return (num_correct / num_samples * 100).item()


def size_profile(model: nn.Module,
                 dummy_input: torch.tensor) -> Tuple[int, int]:
    params = sum(p.numel() for p in model.parameters())
    macs = torchprofile.profile_macs(model, dummy_input)

    return params, macs


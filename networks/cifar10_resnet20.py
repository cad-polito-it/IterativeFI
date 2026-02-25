import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from .mnist_mlp import train_with_early_stopping, test_model
from copy import deepcopy

from torch.quantization import quantize_fx


def conv3x3(in_channels, out_channels, stride=1):
    """
    return 3x3 Conv2d
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    """
    Basic Residual Block
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet for CIFAR (ResNet20 if layers=[3,3,3])
    """
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16

        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def main():
    root = "../../data"

    train_ratio = 0.8
    validation_ratio = 0.2

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train
    )

    dataset_size = len(train_dataset)
    train_size = int(train_ratio * dataset_size)
    validation_size = dataset_size - train_size

    train_dataset, validation_dataset = random_split(
        train_dataset, [train_size, validation_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=128,
                              shuffle=True, num_workers=2)

    validation_loader = DataLoader(validation_dataset, batch_size=128,
                                   shuffle=False)

    test_dataset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test
    )

    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(ResidualBlock, [3, 3, 3]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1
    )

    model, history = train_with_early_stopping(
        model, train_loader, validation_loader,
        optimizer, criterion, device,
        scheduler=scheduler,
        save_path="best_cifar_resnet20.pt"
    )

    test_model(model, test_loader, device,
               load_path="./weights/best_cifar_resnet20.pt")

    ## FX GRAPH QUANTIZATION
    backend = "fbgemm"

    model.cpu()
    model.eval()
    m = deepcopy(model)
    m.eval()

    qconfig_dict = {
        "": torch.quantization.get_default_qconfig(backend)
    }

    calibration_loader = DataLoader(
        Subset(train_dataset, range(100)),
        batch_size=128, shuffle=False
    )

    model_prepared = quantize_fx.prepare_fx(
        m, qconfig_dict,
        example_inputs=next(iter(calibration_loader))[0]
    )

    with torch.inference_mode():
        for inputs, _ in calibration_loader:
            model_prepared(inputs)

    model_quantized = quantize_fx.convert_fx(model_prepared)

    for name, module in model_quantized.named_modules():
        if hasattr(module, "weight"):
            w = module.weight()
            print(f"Layer: {name}, shape: {w.shape}, dtype: {w.dtype}")
            w_int = w.int_repr()
            print(f"Layer {name} int weights:\n{w_int}")


def get_cifar10_resnet20(train_loader, test_loader,
                         load_path="./weights/best_cifar_resnet20.pt",
                         quantized=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(ResidualBlock, [3, 3, 3]).to(device)

    test_model(model, test_loader, device,
               load_path=load_path)

    if quantized:

        backend = "fbgemm"

        model.cpu()
        model.eval()

        m = deepcopy(model)
        m.eval()

        qconfig_dict = {
            "": torch.quantization.get_default_qconfig(backend)
        }

        train_dataset = train_loader.dataset

        calibration_loader = DataLoader(
            Subset(train_dataset, range(100)),
            batch_size=128, shuffle=False
        )

        model_prepared = quantize_fx.prepare_fx(
            m, qconfig_dict,
            example_inputs=next(iter(calibration_loader))[0]
        )

        with torch.inference_mode():
            for inputs, _ in calibration_loader:
                model_prepared(inputs)

        model_quantized = quantize_fx.convert_fx(model_prepared)

        for name, module in model_quantized.named_modules():
            if hasattr(module, "weight"):
                w = module.weight()
                print(f"Layer: {name}, shape: {w.shape}, dtype: {w.dtype}")
                w_int = w.int_repr()
                print(f"Layer {name} int weights:\n{w_int}")

        return model_quantized

    return model


if __name__ == "__main__":
    main()

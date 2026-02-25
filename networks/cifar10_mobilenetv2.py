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


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
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
        root=root, train=True,
        download=True, transform=transform_train
    )

    dataset_size = len(train_dataset)
    train_size = int(train_ratio * dataset_size)
    validation_size = dataset_size - train_size

    train_dataset, validation_dataset = random_split(
        train_dataset, [train_size, validation_size]
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=128,
                              shuffle=True,
                              num_workers=2)

    validation_loader = DataLoader(validation_dataset,
                                   batch_size=128,
                                   shuffle=False)

    test_dataset = torchvision.datasets.CIFAR10(
        root=root, train=False,
        download=True, transform=transform_test
    )

    test_loader = DataLoader(test_dataset,
                             batch_size=128,
                             shuffle=False,
                             num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MobileNetV2().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=5e-4)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=30, gamma=0.1
    )

    model, history = train_with_early_stopping(
        model, train_loader, validation_loader,
        optimizer, criterion, device,
        scheduler=scheduler,
        save_path="best_cifar_mobilenetv2.pt"
    )

    test_model(model, test_loader, device,
               load_path="./weights/best_cifar_mobilenetv2.pt")

    # ----------------- FX QUANTIZATION -----------------

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

    example_inputs = next(iter(calibration_loader))[0]

    model_prepared = quantize_fx.prepare_fx(
        m, qconfig_dict,
        example_inputs=example_inputs
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


def get_cifar10_mobilenetv2(train_loader, test_loader,
                            load_path="./weights/best_cifar_mobilenetv2.pt",
                            quantized=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MobileNetV2().to(device)

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

        example_inputs = next(iter(calibration_loader))[0]

        model_prepared = quantize_fx.prepare_fx(
            m, qconfig_dict,
            example_inputs=example_inputs
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

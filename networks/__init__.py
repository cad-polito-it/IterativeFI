from .cifar10_resnet18 import get_cifar10_resnet18
from .cifar10_resnet20 import get_cifar10_resnet20
from .cifar10_mobilenetv2 import get_cifar10_mobilenetv2
from .mnist_cnn import get_mnist_cnn
from .mnist_mlp import get_mnist_mlp, test_model, train_with_early_stopping
from .banknote_mlp import get_banknote_mlp

networks = {
    'cifar10_resnet18': get_cifar10_resnet18,
    'cifar10_resnet20': get_cifar10_resnet20,
    'cifar10_mobilenetv2': get_cifar10_mobilenetv2,
    "mnist_cnn": get_mnist_cnn,
    'mnist_mlp': get_mnist_mlp,
    'banknote_mlp': get_banknote_mlp
}

def get_network(name, **kwargs):
    """Network"""
    return networks[name.lower()](**kwargs)

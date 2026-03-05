from .googlenet_cifar10 import get_googlenet
from .mnist_cnn import get_mnist_cnn
from .mnist_mlp import get_mnist_mlp, test_model, train_with_early_stopping
from .banknote_mlp import get_banknote_mlp

networks = {
    "mnist_cnn": get_mnist_cnn,
    'mnist_mlp': get_mnist_mlp,
    'banknote_mlp': get_banknote_mlp,
    "googlenet": get_googlenet,

}

def get_network(name, **kwargs):
    """Network"""
    return networks[name.lower()](**kwargs)

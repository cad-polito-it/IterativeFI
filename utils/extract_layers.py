
import argparse
import torch

parser = argparse.ArgumentParser(
        description="Extract Conv and Linear layer names from a PyTorch model."
    )

parser.add_argument(
    "-weights_path",
    type=str,
    help="Path to the saved PyTorch model (.pt or .pth)",
)

def load_state_dict(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    return ckpt


def get_layer_names(state_dict):
    layer_names = set()

    for key in state_dict.keys():
        # rimuove .weight / .bias / ecc.
        layer_name = key.rsplit(".", 1)[0]
        layer_names.add(layer_name)

    return sorted(layer_names)


def main(args):
    state_dict = load_state_dict(args.weights_path)
    layers = get_layer_names(state_dict)
    print("\n".join(layers))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
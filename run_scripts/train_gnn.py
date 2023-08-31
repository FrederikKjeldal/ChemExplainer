import yaml
import argparse

from gnn.train import train_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configs/example.yaml')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    kwargs = args.__dict__

    with open(kwargs['config']) as f:
        config = yaml.safe_load(f)

    train_model(config)
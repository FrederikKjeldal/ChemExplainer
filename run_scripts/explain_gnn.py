import argparse

import yaml

from explain.explain import explain_dataset, explain_smiles


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--smiles", default="CC(=O)OC1=CC=CC=C1C(=O)O")
    parser.add_argument("--dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    kwargs = args.__dict__

    with open(kwargs["config"]) as f:
        config = yaml.safe_load(f)

    if kwargs["dataset"] is None:
        explain_smiles(kwargs["smiles"], config)
    else:
        explain_dataset(kwargs["dataset"], config)

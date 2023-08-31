# Chem GNN Explainers

A repository for various GNN explainers for use in chemistry.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To install this module for development:

```
python setup.py develop
```

## Data

Datasets should be stored in the data folder. Your data should be stored in a csv filed called `data.csv`. The data should be in an overall folder, with a `raw/` subfolder where you store your csv file. A `processed/` subfolder containing the graphs will be created when you train the model.

Example:
```
data/MUTAG/raw/data.csv
```

## Configs

An example config is shown in `configs/example.config`.


## Training and testing

A model can be trained from the example config by running the command: 
```
python run_scripts/train_gnn.py --config configs/example.yaml
```

Similarly you can test the trained model by running:
```
python run_scripts/test_gnn.py --config configs/example.yaml
```
or 

```
python run_scripts/test_gnn.py --config runs/example/config.yaml 
```

## Explaining predictions

You can explain your predictions for a single smiles string or an entire dataset.

For explaining a single smiles:
```
python run_scripts/explain_gnn.py --config runs/example/config.yaml --smiles O=[N+]\([O-]\)c1cc\(C\(O\)=Nc2cccc\(Br\)c2\)cs1
```

For explaining a dataset:
```
python run_scripts/explain_gnn.py --config runs/example/config.yaml --dataset data/MUTAG/raw/data.csv
```
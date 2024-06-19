import os

import dgl
import numpy as np
import pandas as pd
import torch
from dgl import DGLGraph, load_graphs, save_graphs
from rdkit import Chem
from tqdm import tqdm

from data.features import atom_features, bond_features, etype_features


def load_data(config):
    # check if processed data exists
    if not os.path.exists(config["dataset_folder"] + "processed/data.bin"):
        process_csv(config)

    return load_processed_data(config)


def process_csv(config):
    csv_data_path = config["dataset_folder"] + "raw/data.csv"

    data = pd.read_csv(csv_data_path)

    # process dataset to graphs and labels
    graph_list = []
    label_list = []
    for i, (smiles, label) in enumerate(tqdm(zip(data["smiles"], data["label"]))):
        graph = construct_dgl_graph_from_smiles(smiles)

        graph_list.append(graph)
        if label == 0:
            idx1, idx2 = 1, 0
        if label == 1:
            idx1, idx2 = 0, 1
        label_list.append([idx1, idx2])

    labels = {"labels": torch.tensor(label_list)}

    save_graphs(config["dataset_folder"] + "processed/data.bin", graph_list, labels)

    return


def construct_dgl_graph_from_smiles(smiles, dtype=torch.float32):
    g = DGLGraph()

    # Add nodes
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    atoms_feature_all = []
    for i, atom in enumerate(mol.GetAtoms()):
        atom_feature = atom_features(atom)
        atoms_feature_all.append(atom_feature)
    g.ndata["node"] = torch.tensor(atoms_feature_all, dtype=dtype)

    # Add edges
    src_list = []
    dst_list = []
    etype_feature_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        etype_feature = bond_features(bond)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
        etype_feature_all.append(etype_feature)
        etype_feature_all.append(etype_feature)

    g.add_edges(src_list, dst_list)
    g.edata["edge"] = torch.tensor(etype_feature_all, dtype=dtype)

    if len(smiles) == 1:
        g = dgl.add_self_loop(g)

    return g


# load data function
def load_processed_data(config):
    processed_data_path = config["dataset_folder"] + "processed/data.bin"
    graphs, detailed_information = load_graphs(processed_data_path)
    labels = detailed_information["labels"]

    # get train, valid, and test index
    train_split = config["train_split"]
    valid_split = config["valid_split"]
    test_split = config["test_split"]
    seed = config["seed"]

    assert (
        train_split + valid_split + test_split <= 1.0
    ), "train, valid, and test size is larger than data"

    N_data = len(labels)
    random_state = np.random.RandomState(seed=seed)
    idx = random_state.permutation(np.arange(N_data))

    train_size = int(N_data * train_split)
    valid_size = int(N_data * valid_split)
    test_size = int(N_data * test_split)
    train_index = idx[:train_size]
    valid_index = idx[train_size : train_size + valid_size]
    test_index = idx[train_size + valid_size : train_size + valid_size + test_size]

    # split data
    train_set = []
    val_set = []
    test_set = []
    for i in train_index:
        molecule = [graphs[i], labels[i]]
        train_set.append(molecule)

    for i in valid_index:
        molecule = [graphs[i], labels[i]]
        val_set.append(molecule)

    for i in test_index:
        molecule = [graphs[i], labels[i]]
        test_set.append(molecule)

    return train_set, val_set, test_set


# collate function
def collate_molgraphs(data):
    graph, labels = map(list, zip(*data))
    graph = dgl.batch(graph)
    labels = torch.stack(labels)  # , dtype=torch.float32)
    return graph, labels

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data.data import collate_molgraphs, construct_dgl_graph_from_smiles, load_data
from explain.chemsubgraphx import ChemSubgraphX
from explain.gnnexplainer import GNNExplainer
from explain.pgexplainer import PGExplainer
from explain.sme import SME
from explain.subgraphx import SubgraphX
from explain.utils import create_dirs, explain_edges_to_atoms
from explain.visualize import save_explanation_svg, save_smiles_svg
from gnn.utils import model_from_config


def explain_smiles(smiles, config):
    # explanation directory
    save_dir = "explanations/" + config["run_name"]

    # create dirs
    create_dirs(save_dir, config["num_classes"], smiles)

    # dump config
    with open(f"{save_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)

    # save smiles string to explain
    save_smiles_svg(smiles, f"{save_dir}/{smiles}/{smiles}.svg")

    # init model from config
    model = model_from_config(config)
    model_dir = f"runs/{config['run_name']}/checkpoints"
    model.load_state_dict(torch.load(f"{model_dir}/best_model.pt")["model_state_dict"])
    model.eval()

    # get graph
    graph = construct_dgl_graph_from_smiles(smiles)

    # evaulate model
    probs = model(graph, graph.ndata["node"], graph.edata["edge"]).softmax(dim=-1)

    # PGExplainer
    # explain_with_pgexplainer(smiles, graph, model, config)

    # GNNExplainer
    # explain_with _gnnexplainer(smiles, graph, model, config)

    for target_class in range(config["num_classes"]):
        # save prediction
        with open(f"{save_dir}/{smiles}/class_{target_class}/prediction.txt", "w") as f:
            f.write(str(probs[:, target_class].item()))

        # SubgraphX
        # explain_with_subgraphx(smiles, graph, model, target_class, config)

        # SME
        explain_with_sme(smiles, graph, model, target_class, config)

        # ChemSubgraphX
        explain_with_chemsubgraphx(smiles, graph, model, target_class, config)

    return


def explain_dataset(dataset_path, config):
    # explanation directory
    save_dir = "explanations/" + config["run_name"]

    # create dirs
    create_dirs(save_dir)

    # dump config
    with open(f"{save_dir}/config.yaml", "w") as f:
        yaml.dump(config, f)

    # init model from config
    model = model_from_config(config)
    model_dir = f"runs/{config['run_name']}/checkpoints"
    model.load_state_dict(torch.load(f"{model_dir}/best_model.pt")["model_state_dict"])
    model.eval()

    # datasets
    data = pd.read_csv(dataset_path)

    # are there labels in the data?
    try:
        labels_exist = True
        labels = data["label"]
    except KeyError:
        labels_exist = False

    for i, smiles in enumerate(data["smiles"]):
        # create dirs
        create_dirs(save_dir, config["num_classes"], smiles)

        # save smiles string to explain
        save_smiles_svg(smiles, f"{save_dir}/{smiles}/{smiles}.svg")
        if labels_exist:
            # save label
            with open(f"{save_dir}/{smiles}/label.txt", "w") as f:
                f.write(str(labels[i]))

        # get different graphs needed
        graph = construct_dgl_graph_from_smiles(smiles)

        # evaulate model
        probs = model(graph, graph.ndata["node"], graph.edata["edge"]).softmax(dim=-1)

        # PGExplainer
        # explain_with_pgexplainer(data, i, smiles, graph, model, config)

        # GNNExplainer
        explain_with_gnnexplainer(smiles, graph, model, config)

        for target_class in range(config["num_classes"]):
            # save prediction
            with open(f"{save_dir}/{smiles}/class_{target_class}/prediction.txt", "w") as f:
                f.write(str(probs[:, target_class].item()))

            # SubgraphX
            explain_with_subgraphx(smiles, graph, model, target_class, config)

            # SME
            explain_with_sme(smiles, graph, model, target_class, config)

            # ChemSubgraphX
            explain_with_chemsubgraphx(smiles, graph, model, target_class, config)

    return


def explain_with_pgexplainer(data, i, smiles, graph_to_explain, model, config):
    if i != 0:
        return

    # explanation directory
    save_dir = "explanations/" + config["run_name"]

    # initialize the explainer
    explainer = PGExplainer(model, config["hidden_size"], num_hops=config["gnn_layers"])

    # get data
    train_data, _, _ = load_data(config)
    batch_size = config["batch_size"]
    train_loader = DataLoader(
        dataset=train_data, batch_size=batch_size, collate_fn=collate_molgraphs
    )

    # train the explainer
    init_tmp, final_tmp = 5.0, 1.0
    optimizer_exp = torch.optim.Adam(explainer.parameters(), lr=0.01)
    for epoch in range(20):
        tmp = float(init_tmp * np.power(final_tmp / init_tmp, epoch / 20))
        for batch_id, batch_data in enumerate(train_loader):
            graph, labels = batch_data
            loss = explainer.train_step(graph, graph.ndata["node"], graph.edata["edge"], tmp)
            optimizer_exp.zero_grad()
            loss.backward()
            optimizer_exp.step()

    print("Finished PGExplainer training")

    for i, smiles in enumerate(data["smiles"]):
        if i % 100 == 0:
            print(i / len(data["smiles"]))
        # get different graphs needed
        graph = construct_dgl_graph_from_smiles(smiles)

        # explain the prediction for graph
        probs, edge_weight = explainer.explain_graph(
            graph, graph.ndata["node"], graph.edata["edge"]
        )

        # visualize explained graph
        max_inds = edge_weight.detach().numpy().argsort()[-5:][::-1]
        new_edges = np.zeros_like(edge_weight.detach().numpy())
        new_edges[max_inds] = 1.0
        atoms = explain_edges_to_atoms(smiles, new_edges)
        save_explanation_svg(smiles, atoms, f"{save_dir}/{smiles}/pgexplainer.svg")

    return


def explain_with_gnnexplainer(smiles, graph_to_explain, model, config):
    # explanation directory
    save_dir = "explanations/" + config["run_name"]

    # initialize the explainer
    explainer = GNNExplainer(model, num_hops=config["gnn_layers"])

    # explain the prediction for graph
    feat_mask, edge_mask = explainer.explain_graph(
        graph_to_explain, graph_to_explain.ndata["node"], graph_to_explain.edata["edge"]
    )

    # visualize explained graph
    atoms = explain_edges_to_atoms(smiles, edge_mask)
    save_explanation_svg(smiles, atoms, f"{save_dir}/{smiles}/gnnexplainer.svg")

    return


def explain_with_subgraphx(smiles, graph_to_explain, model, target_class, config):
    # explanation directory
    save_dir = "explanations/" + config["run_name"]

    # initialize the explainer
    explainer = SubgraphX(model, num_hops=config["gnn_layers"])

    # explain the prediction for graph
    nodes_explain = explainer.explain_graph(
        graph_to_explain,
        graph_to_explain.ndata["node"],
        graph_to_explain.edata["edge"],
        target_class=target_class,
    )

    best_immediate_reward = 0
    best_nodes = torch.tensor([])
    for node_value in nodes_explain:
        reward = node_value[1]
        nodes = node_value[0]

        if reward > best_immediate_reward:
            best_immediate_reward = reward
            best_nodes = nodes

    # visualize explained graph
    atoms = best_nodes.tolist()
    save_explanation_svg(smiles, atoms, f"{save_dir}/{smiles}/class_{target_class}/subgraphx.svg")

    return


def explain_with_sme(smiles, graph_to_explain, model, target_class, config):
    # explanation directory
    save_dir = "explanations/" + config["run_name"]

    # initialize the explainer
    explainer = SME(model)

    # explain the prediction for graph
    best_immediate_reward = 0
    best_nodes = []
    for mask_type in ["fg", "murcko", "brics"]:
        nodes_explain = explainer.explain_graph(
            smiles,
            graph_to_explain,
            graph_to_explain.ndata["node"],
            graph_to_explain.edata["edge"],
            target_class=target_class,
            mask_type=mask_type,
        )

        for node_value in nodes_explain:
            reward = node_value[1]
            nodes = node_value[0]

            if reward > best_immediate_reward:
                best_immediate_reward = reward
                best_nodes = nodes

    # visualize explained graph
    atoms = best_nodes
    save_explanation_svg(smiles, atoms, f"{save_dir}/{smiles}/class_{target_class}/sme.svg")

    return


def explain_with_chemsubgraphx(smiles, graph_to_explain, model, target_class, config):
    # explanation directory
    save_dir = "explanations/" + config["run_name"]

    # initialize the explainer
    explainer = ChemSubgraphX(model, num_hops=config["gnn_layers"])

    # explain the prediction for graph
    nodes_explain = explainer.explain_graph(
        smiles,
        graph_to_explain,
        graph_to_explain.ndata["node"],
        graph_to_explain.edata["edge"],
        target_class=target_class,
    )

    best_immediate_reward = 0
    best_nodes = [[]]
    for node_value in nodes_explain:
        reward = node_value[1]
        nodes = node_value[0]

        if reward > best_immediate_reward:
            best_immediate_reward = reward
            best_nodes = nodes

    # visualize explained graph
    atoms = [atom for subgroup in best_nodes for atom in subgroup]
    save_explanation_svg(
        smiles, atoms, f"{save_dir}/{smiles}/class_{target_class}/chemsubgraphx.svg"
    )

    return

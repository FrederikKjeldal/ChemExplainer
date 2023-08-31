import yaml
import dgl
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data.data import load_data, collate_molgraphs, construct_dgl_graph_from_smiles
from gnn.utils import model_from_config
from explain.utils import explain_edges_to_atoms, create_dirs
from explain.pgexplainer import PGExplainer
from explain.gnnexplainer import GNNExplainer
from explain.subgraphx import SubgraphX
from explain.sme import SME
from explain.chemsubgraphx import ChemSubgraphX
from explain.visualize import save_smiles_svg, save_explanation_svg

def explain_smiles(smiles, config):
    # explanation directory
    save_dir = 'explanations/' + config['run_name']
    
    # create dirs
    create_dirs(save_dir, config['num_classes'], smiles)

    # dump config
    with open(f'{save_dir}/config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # save smiles string to explain
    save_smiles_svg(smiles, f'{save_dir}/{smiles}/{smiles}.svg')

    # init model from config
    model = model_from_config(config)
    model_dir = f"runs/{config['run_name']}/checkpoints"
    model.load_state_dict(torch.load(f'{model_dir}/best_model.pt')['model_state_dict'])
    model.eval()

    # get graph
    graph = construct_dgl_graph_from_smiles(smiles)

    # evaulate model
    probs = model(graph, graph.ndata['node'], graph.edata['edge']).softmax(dim=-1)

    # PGExplainer
    # explain_with_pgexplainer(smiles, graph, model, config)

    # GNNExplainer
    # explain_with _gnnexplainer(smiles, graph, model, config)

    for target_class in range(config['num_classes']):
        # save prediction
        with open(f'{save_dir}/{smiles}/class_{target_class}/prediction.txt', 'w') as f:
            f.write(str(probs[:, target_class].item()))

        # SubgraphX
        explain_with_subgraphx(smiles, graph, model, target_class, config)

        # SME
        explain_with_sme(smiles, graph, model, target_class, config)

        # ChemSubgraphX
        explain_with_chemsubgraphx(smiles, graph, model, target_class, config)

    return

def explain_dataset(dataset_path, config):
    # explanation directory
    save_dir = 'explanations/' + config['run_name']
    
    # create dirs
    create_dirs(save_dir)

    # dump config
    with open(f'{save_dir}/config.yaml', 'w') as f:
        yaml.dump(config, f)

    # init model from config
    model = model_from_config(config)
    model_dir = f"runs/{config['run_name']}/checkpoints"
    model.load_state_dict(torch.load(f'{model_dir}/best_model.pt')['model_state_dict'])
    model.eval()
    
    # datasets
    data = pd.read_csv(dataset_path)

    for smiles in data['smiles']:
        # create dirs
        create_dirs(save_dir, config['num_classes'], smiles)
        
        # get different graphs needed 
        graph = construct_dgl_graph_from_smiles(smiles)

        # evaulate model
        probs = model(graph, graph.ndata['node'], graph.edata['edge']).softmax(dim=-1)

        # PGExplainer
        # explain_with_pgexplainer(smiles, graph, model, config)

        # GNNExplainer
        # explain_with_gnnexplainer(smiles, graph, model, config)

        for target_class in range(config['num_classes']):
            # save prediction
            with open(f'{save_dir}/{smiles}/class_{target_class}/prediction.txt', 'w') as f:
                f.write(str(probs[:, target_class].item()))

            # SubgraphX
            explain_with_subgraphx(smiles, graph, model, target_class, config)

            # SME
            explain_with_sme(smiles, graph, model, target_class, config)

            # ChemSubgraphX
            explain_with_chemsubgraphx(smiles, graph, model, target_class, config)
    
    return

def explain_with_pgexplainer(smiles, graph_to_explain, model, config):
    # explanation directory
    save_dir = 'explanations/' + config['run_name']

    # initialize the explainer
    explainer = PGExplainer(model, config['hidden_size'], num_hops=config['gnn_layers'])

    # get data
    train_data, _, _ = load_data(config)
    batch_size = config['batch_size']
    train_loader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                collate_fn=collate_molgraphs)

    # train the explainer
    init_tmp, final_tmp = 5.0, 1.0
    optimizer_exp = torch.optim.Adam(explainer.parameters(), lr=0.01)
    for epoch in range(20):
        tmp = float(init_tmp * np.power(final_tmp / init_tmp, epoch / 20))
        for batch_id, batch_data in enumerate(train_loader):
            graph, labels = batch_data
            loss = explainer.train_step(graph, graph.ndata['node'], graph.edata['edge'], tmp)
            optimizer_exp.zero_grad()
            loss.backward()
            optimizer_exp.step()

    # explain the prediction for graph
    probs, edge_weight = explainer.explain_graph(graph_to_explain, graph_to_explain.ndata['node'], graph_to_explain.edata['edge'])

    # visualize explained graph
    atoms = explain_edges_to_atoms(smiles, edge_weight)
    save_explanation_svg(smiles, atoms, f'{save_dir}/{smiles}/gnnexplainer.svg')

    return

def explain_with_gnnexplainer(smiles, graph_to_explain, model, config):
    # explanation directory
    save_dir = 'explanations/' + config['run_name']

    # initialize the explainer
    explainer = GNNExplainer(model, num_hops=config['gnn_layers'])

    # explain the prediction for graph
    feat_mask, edge_mask = explainer.explain_graph(graph_to_explain, graph_to_explain.ndata['node'], graph_to_explain.edata['edge'])

    # visualize explained graph
    atoms = explain_edges_to_atoms(smiles, edge_mask)
    save_explanation_svg(smiles, atoms, f'{save_dir}/{smiles}/gnnexplainer.svg')

    return

def explain_with_subgraphx(smiles, graph_to_explain, model, target_class, config):
    # explanation directory
    save_dir = 'explanations/' + config['run_name']

    # initialize the explainer
    explainer = SubgraphX(model, num_hops=config['gnn_layers'])

    # explain the prediction for graph
    nodes = explainer.explain_graph(graph_to_explain, graph_to_explain.ndata['node'], graph_to_explain.edata['edge'], target_class=target_class)

    # visualize explained graph
    atoms = nodes.tolist()
    save_explanation_svg(smiles, atoms, f'{save_dir}/{smiles}/class_{target_class}/subgraphx.svg')

    return

def explain_with_sme(smiles, graph_to_explain, model, target_class, config):
    # explanation directory
    save_dir = 'explanations/' + config['run_name']

    # initialize the explainer
    explainer = SME(model)

    # explain the prediction for graph
    best_immediate_reward = float("-inf")
    for mask_type in ['fg', 'murcko', 'brics', 'combination']:
        nodes_explain = explainer.explain_graph(smiles, graph_to_explain, graph_to_explain.ndata['node'], graph_to_explain.edata['edge'], target_class=target_class, mask_type=mask_type)

        for node_value in nodes_explain:
            reward = node_value[1]
            nodes = node_value[0]

            if reward > best_immediate_reward:
                best_immediate_reward = reward
                best_nodes = nodes

    # visualize explained graph
    atoms = best_nodes
    save_explanation_svg(smiles, atoms, f'{save_dir}/{smiles}/class_{target_class}/sme.svg')

    return

def explain_with_chemsubgraphx(smiles, graph_to_explain, model, target_class, config):
    # explanation directory
    save_dir = 'explanations/' + config['run_name']

    # initialize the explainer
    explainer = ChemSubgraphX(model, num_hops=config['gnn_layers'])

    # explain the prediction for graph
    nodes_explain = explainer.explain_graph(smiles, graph_to_explain, graph_to_explain.ndata['node'], graph_to_explain.edata['edge'], target_class=target_class)

    best_immediate_reward = float("-inf")
    for node_value in nodes_explain:
        reward = node_value[1]
        nodes = node_value[0]

        if reward > best_immediate_reward:
            best_immediate_reward = reward
            best_nodes = nodes

    # visualize explained graph
    atoms = [atom for subgroup in best_nodes for atom in subgroup]
    save_explanation_svg(smiles, atoms, f'{save_dir}/{smiles}/class_{target_class}/chemsubgraphx.svg')

    return

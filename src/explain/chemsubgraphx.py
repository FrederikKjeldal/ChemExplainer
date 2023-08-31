from dgl.nn import SubgraphX
from dgl.nn.pytorch.explain.subgraphx import MCTSNode
import networkx as nx
import torch
import numpy as np
from dgl.base import NID
from dgl.convert import to_networkx
from dgl.subgraph import node_subgraph
from dgl.transforms.functional import remove_nodes

from explain.utils import generate_chem_subgraphs


class ChemSubgraphX(SubgraphX):
    def __init__(
            self,
            model,
            num_hops,
            coef=10.0,
            high2low=True,
            num_child=12,
            num_rollouts=20,
            node_min=1,
            shapley_steps=100,
            log=False,
    ):
        super(ChemSubgraphX, self).__init__(
            model,
            num_hops,
            coef,
            high2low,
            num_child,
            num_rollouts,
            node_min,
            shapley_steps,
            log
        )

    def shapley(self, subgraph_nodes):
        num_nodes = self.simple_graph.num_nodes()
        real_graph_num_nodes = self.graph.num_nodes()
        graph_nodes = self.simple_graph.nodes().tolist()
        subgraph_nodes = subgraph_nodes.tolist()

        # Obtain neighboring nodes of the subgraph g_i, P'.
        local_region = subgraph_nodes
        for _ in range(self.num_hops - 1):
            in_neighbors, _ = self.simple_graph.in_edges(local_region)
            _, out_neighbors = self.simple_graph.out_edges(local_region)
            neighbors = torch.cat([in_neighbors, out_neighbors]).tolist()
            local_region = list(set(local_region + neighbors))

        split_point = num_nodes
        coalition_space = list(set(local_region) - set(subgraph_nodes)) + [
            split_point
        ]

        marginal_contributions = []
        device = self.node_feat.device
        for _ in range(self.shapley_steps):
            permuted_space = np.random.permutation(coalition_space)
            split_idx = int(np.where(permuted_space == split_point)[0])

            selected_nodes = permuted_space[:split_idx]

            # Mask for coalition set S_i
            exclude_mask = torch.ones(num_nodes)
            exclude_mask[local_region] = 0.0
            exclude_mask[selected_nodes] = 1.0

            # Mask for set S_i and g_i
            include_mask = exclude_mask.clone()
            include_mask[subgraph_nodes] = 1.0

            # Get real graph masks
            real_exclude_mask = torch.ones(real_graph_num_nodes)
            real_include_mask = torch.ones(real_graph_num_nodes)
            for i, (e_mask, i_mask) in enumerate(zip(exclude_mask, include_mask)):
                node = graph_nodes[i]
                for real_node in self.subgroups[node]:
                    real_exclude_mask[real_node] = e_mask
                    real_include_mask[real_node] = i_mask

            exclude_node_feat = self.node_feat * real_exclude_mask.unsqueeze(1).to(device)
            include_node_feat = self.node_feat * real_include_mask.unsqueeze(1).to(device)

            with torch.no_grad():
                exclude_probs = self.model(
                    self.graph, exclude_node_feat, self.edge_feat, **self.kwargs
                ).softmax(dim=-1)
                exclude_value = exclude_probs[:, self.target_class]
                include_probs = self.model(
                    self.graph, include_node_feat, self.edge_feat, **self.kwargs
                ).softmax(dim=-1)
                include_value = include_probs[:, self.target_class]
            marginal_contributions.append(include_value - exclude_value)

        return torch.cat(marginal_contributions).mean().item()
    
    def get_mcts_children(self, mcts_node):
        r"""Get the children of the MCTS node for the search.

        Parameters
        ----------
        mcts_node : MCTSNode
            Node in MCTS

        Returns
        -------
        list
            Children nodes after pruning
        """
        if len(mcts_node.children) > 0:
            return mcts_node.children

        subg = node_subgraph(self.simple_graph, mcts_node.nodes)
        node_degrees = subg.out_degrees() + subg.in_degrees()
        k = min(subg.num_nodes(), self.num_child)
        chosen_nodes = torch.topk(
            node_degrees, k, largest=self.high2low
        ).indices

        mcts_children_maps = dict()

        for node in chosen_nodes:
            new_subg = remove_nodes(subg, node.to(subg.idtype), store_ids=True)
            # Get the largest weakly connected component in the subgraph.
            nx_graph = to_networkx(new_subg.cpu())
            largest_cc_nids = list(
                max(nx.weakly_connected_components(nx_graph), key=len)
            )
            # Map to the original node IDs.
            largest_cc_nids = new_subg.ndata[NID][largest_cc_nids].long()
            largest_cc_nids = subg.ndata[NID][largest_cc_nids].sort().values
            if str(largest_cc_nids) not in self.mcts_node_maps:
                child_mcts_node = MCTSNode(largest_cc_nids)
                self.mcts_node_maps[str(child_mcts_node)] = child_mcts_node
            else:
                child_mcts_node = self.mcts_node_maps[str(largest_cc_nids)]

            if str(child_mcts_node) not in mcts_children_maps:
                mcts_children_maps[str(child_mcts_node)] = child_mcts_node

        mcts_node.children = list(mcts_children_maps.values())
        for child_mcts_node in mcts_node.children:
            if child_mcts_node.immediate_reward == 0:
                child_mcts_node.immediate_reward = self.shapley(
                    child_mcts_node.nodes
                )

        return mcts_node.children
    
    def generate_simplified_graph(self, smiles):
        simple_graph, subgroups = generate_chem_subgraphs(smiles)

        self.simple_graph = simple_graph
        self.subgroups = subgroups
    
    def explain_graph(self, smiles, graph, node_feat, edge_feat, target_class, **kwargs):
        self.model.eval()
        assert (
            graph.num_nodes() >= self.node_min
        ), f"The number of nodes in the\
            graph {graph.num_nodes()} should be bigger than or equal to {self.node_min}."

        self.graph = graph
        self.node_feat = node_feat
        self.edge_feat = edge_feat
        self.target_class = target_class
        self.kwargs = kwargs

        self.generate_simplified_graph(smiles)

        # book all nodes in MCTS
        self.mcts_node_maps = dict()

        root = MCTSNode(self.simple_graph.nodes())
        self.mcts_node_maps[str(root)] = root

        for i in range(self.num_rollouts):
            if self.log:
                print(
                    f"Rollout {i}/{self.num_rollouts}, \
                    {len(self.mcts_node_maps)} subgraphs have been explored."
                )
            self.mcts_rollout(root)

        best_leaf = None
        best_immediate_reward = float("-inf")
        nodes_values = []
        for mcts_node in self.mcts_node_maps.values():
            nodes = []
            for simple_node in mcts_node.nodes:
                nodes.append(self.subgroups[simple_node])
            nodes_values.append((nodes, mcts_node.immediate_reward))

            if mcts_node.immediate_reward > best_immediate_reward:
                best_leaf = mcts_node
                best_immediate_reward = best_leaf.immediate_reward

        return nodes_values
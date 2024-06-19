import torch
import torch.nn as nn

from explain.utils import (
    generate_pair_combinations,
    sme_brics_masks,
    sme_fg_masks,
    sme_murcko_masks,
)


class SME(nn.Module):
    def __init__(
        self,
        model,
        log=False,
    ):
        super().__init__()
        self.model = model
        self.log = log

    def attributions(self, masks):
        attribution_values = []

        for mask in masks:
            with torch.no_grad():
                y = self.model(self.graph, self.node_feat, self.edge_feat).softmax(dim=-1)[
                    :, self.target_class
                ]
                y_mask = self.model(self.graph, self.node_feat, self.edge_feat, mask=mask).softmax(
                    dim=-1
                )[:, self.target_class]

            attribution_values.append((y - y_mask).item())

        return attribution_values

    def explain_graph(
        self, smiles, graph, node_feat, edge_feat, target_class, mask_type="fg", **kwargs
    ):
        self.model.eval()

        self.graph = graph
        self.node_feat = node_feat
        self.edge_feat = edge_feat
        self.target_class = target_class
        self.kwargs = kwargs

        if mask_type == "fg":
            atoms, masks = sme_fg_masks(smiles)
        elif mask_type == "murcko":
            atoms, masks = sme_murcko_masks(smiles)
        elif mask_type == "brics":
            atoms, masks = sme_brics_masks(smiles)
        elif mask_type == "combination":
            atoms, masks = generate_pair_combinations(smiles)
        else:
            raise NotImplementedError

        attributions = self.attributions(masks)

        nodes_values = []
        for nodes, values in zip(atoms, attributions):
            nodes_values.append((nodes, values))

        return nodes_values

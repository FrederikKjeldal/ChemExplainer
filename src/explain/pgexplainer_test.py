import dgl
import numpy as np
import torch as th
import torch.nn as nn
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv, PGExplainer


# Define the model
class Model(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.conv = GraphConv(in_feats, out_feats)
        self.fc = nn.Linear(out_feats, out_feats)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, g, h, embed=False, edge_weight=None):
        h = self.conv(g, h, edge_weight=edge_weight)

        if embed:
            return h

        with g.local_scope():
            g.ndata["h"] = h
            hg = dgl.mean_nodes(g, "h")
            return self.fc(hg)


# Load dataset
data = GINDataset("MUTAG", self_loop=True)
dataloader = GraphDataLoader(data, batch_size=64, shuffle=True)

# Train the model
feat_size = data[0][0].ndata["attr"].shape[1]
model = Model(feat_size, data.gclasses)
criterion = nn.CrossEntropyLoss()
optimizer = th.optim.Adam(model.parameters(), lr=1e-2)
for bg, labels in dataloader:
    preds = model(bg, bg.ndata["attr"])
    loss = criterion(preds, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Initialize the explainer
explainer = PGExplainer(model, data.gclasses)

# Train the explainer
# Define explainer temperature parameter
init_tmp, final_tmp = 5.0, 1.0
optimizer_exp = th.optim.Adam(explainer.parameters(), lr=0.01)
for epoch in range(20):
    tmp = float(init_tmp * np.power(final_tmp / init_tmp, epoch / 20))
    for bg, labels in dataloader:
        loss = explainer.train_step(bg, bg.ndata["attr"], tmp)
        optimizer_exp.zero_grad()
        loss.backward()
        optimizer_exp.step()

# Explain the prediction for graph 0
graph, l = data[0]
graph_feat = graph.ndata.pop("attr")
probs, edge_weight = explainer.explain_graph(graph, graph_feat)
print(probs)
max_inds = edge_weight.detach().numpy().argsort()[-5:][::-1]
new_edges = np.zeros_like(edge_weight.detach().numpy())
new_edges[max_inds] = 1.0
print(new_edges)

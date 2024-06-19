import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from data.data import collate_molgraphs, load_data
from gnn.utils import model_from_config, setup_logger


def test_model(config):
    # datasets
    _, _, test_data = load_data(config)
    batch_size = config["batch_size"]
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, collate_fn=collate_molgraphs)

    # logging
    save_dir = f"runs/{config['run_name']}"
    setup_logger(save_dir, log_name="test.log", debug=config["debug"])

    # init model from config
    model_dir = f"runs/{config['run_name']}/checkpoints"
    model = model_from_config(config)
    model.load_state_dict(torch.load(f"{model_dir}/best_model.pt")["model_state_dict"])

    metrics(model, test_loader)

    return


def metrics(model, data_loader):
    logging.info("--------------- TESTING ---------------")

    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            graph, labels = batch_data
            trues = labels.float().numpy()
            preds = np.rint(
                model(graph, graph.ndata["node"], graph.edata["edge"]).softmax(dim=-1).numpy()
            )
            if batch_id == 0:
                y_true = trues
                y_pred = preds
            else:
                y_true = np.concatenate((y_true, trues), axis=0)
                y_pred = np.concatenate((y_pred, preds), axis=0)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    roc_auc = roc_auc_score(y_true, y_pred, average=None)

    logging.info(f"Accuracy score: {accuracy}")
    logging.info(f"F1 score: {f1}")
    logging.info(f"ROC AUC score: {roc_auc}")

    return
